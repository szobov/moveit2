/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, R3 Robotics
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of R3 Robotics nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Sergei Zobov */

#include <moveit/ompl_interface/detail/cache_planning.hpp>

#include <ompl/base/PlannerStatus.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalLazySamples.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/util/Console.h>

#include <algorithm>

namespace ompl_interface
{
CachePlanning::CachePlanning(const ompl::base::SpaceInformationPtr& si) : ompl::base::Planner(si, "CachePlanning")
{
  specs_.approximateSolutions = false;
  specs_.directed = true;
  specs_.recognizedGoal = ompl::base::GOAL_SAMPLEABLE_REGION;

  Planner::declareParam<double>("cache_area", this, &CachePlanning::setCacheArea, &CachePlanning::getCacheArea);
  Planner::declareParam<int>("cache_point_attempts", this, &CachePlanning::setCachePointAttempts,
                             &CachePlanning::getCachePointAttempts, "1:1000");
  Planner::declareParam<int>("cache_trajectory_attempts", this, &CachePlanning::setCacheTrajectoryAttempts,
                             &CachePlanning::getCacheTrajectoryAttempts, "1:1000");
}

CachePlanning::~CachePlanning()
{
  freeMemory();
  freeGoalStates();
}

void CachePlanning::setSeedTrajectory(std::vector<std::vector<double>> seed)
{
  seed_trajectory_ = std::move(seed);
}

bool CachePlanning::validateSeedDimensions()
{
  // copyFromReals consumes one value per value location, which can differ from getStateDimension() (e.g.
  // ModelBasedStateSpace exposes one value location per group variable, mimic variables included)
  const std::size_t dimension = si_->getStateSpace()->getValueLocations().size();
  for (const std::vector<double>& waypoint : seed_trajectory_)
  {
    if (waypoint.size() != dimension)
    {
      OMPL_ERROR("%s: Seed waypoint has %zu values but the state space has %zu value locations. Rejecting the whole "
                 "seed.",
                 getName().c_str(), waypoint.size(), dimension);
      seed_trajectory_.clear();
      return false;
    }
  }
  return true;
}

void CachePlanning::setup()
{
  Planner::setup();
  validateSeedDimensions();

  OMPL_DEBUG("%s: cache_area=%f cache_point_attempts=%d cache_trajectory_attempts=%d seed waypoints=%zu",
             getName().c_str(), cache_area_, cache_point_attempts_, cache_trajectory_attempts_,
             seed_trajectory_.size());
}

void CachePlanning::clear()
{
  Planner::clear();
  freeMemory();
  freeGoalStates();
  // seed_trajectory_ is deliberately kept: clear() is called before every solve attempt and the seed must
  // survive it so the same seed can be reconstructed in an updated scene. The state sampler only depends on
  // the state space and is kept as well.
}

void CachePlanning::freeMemory()
{
  for (Motion* motion : motions_)
  {
    si_->freeState(motion->state);
    delete motion;
  }
  motions_.clear();
}

void CachePlanning::freeGoalStates()
{
  for (ompl::base::State* goal_state : goal_states_)
    si_->freeState(goal_state);
  goal_states_.clear();
}

const ompl::base::State* CachePlanning::getGoalState(std::size_t index,
                                                     const ompl::base::PlannerTerminationCondition& ptc)
{
  const auto* lazy_goal = dynamic_cast<ompl::base::GoalLazySamples*>(pdef_->getGoal().get());
  while (goal_states_.size() <= index)
  {
    // Waiting is correct while no goal state is known yet, and while a lazy goal sampler is still actively
    // producing states (a fast reconstruction can otherwise outpace the sampling thread and miss goal states
    // that appear moments later). Exhausted finite goal samplers keep reporting that they could sample, so a
    // waiting nextGoal(ptc) on them would block until the termination condition fires, and this planner must
    // fail fast instead - hence the non-waiting overload in that case.
    const bool may_wait = goal_states_.empty() || (lazy_goal && lazy_goal->isSampling());
    const ompl::base::State* goal_state = may_wait ? pis_.nextGoal(ptc) : pis_.nextGoal();
    if (!goal_state)
      return nullptr;
    goal_states_.push_back(si_->cloneState(goal_state));
  }
  return goal_states_[index];
}

void CachePlanning::resetTreeToStartMotions()
{
  // Start motions are the only ones without a parent
  const auto first_removed =
      std::partition(motions_.begin(), motions_.end(), [](const Motion* motion) { return motion->parent == nullptr; });
  for (auto it = first_removed; it != motions_.end(); ++it)
  {
    si_->freeState((*it)->state);
    delete *it;
  }
  motions_.erase(first_removed, motions_.end());
}

bool CachePlanning::sampleCacheStateRegion(const std::vector<double>& waypoint, ompl::base::State* state,
                                           const Motion* prev_motion)
{
  si_->getStateSpace()->copyFromReals(state, waypoint);

  // A waypoint that repeats the previous one would create a zero-length edge, so it is skipped instead
  if (si_->equalStates(state, prev_motion->state))
    return false;

  if (si_->checkMotion(prev_motion->state, state))
    return true;

  ompl::base::ScopedState<> sampled_state(si_);
  for (int attempt = 0; attempt < cache_point_attempts_; ++attempt)
  {
    sampler_->sampleUniformNear(sampled_state.get(), state, cache_area_);
    if (si_->checkMotion(prev_motion->state, sampled_state.get()))
    {
      si_->copyState(state, sampled_state.get());
      return true;
    }
  }
  return false;
}

bool CachePlanning::cacheLoop(const ompl::base::PlannerTerminationCondition& ptc, Motion* start_motion)
{
  ompl::base::Goal* goal = pdef_->getGoal().get();
  const bool goal_is_sampleable = goal->hasType(ompl::base::GOAL_SAMPLEABLE_REGION);

  Motion* prev_motion = start_motion;

  for (std::size_t step = 0; step < seed_trajectory_.size() && !ptc; ++step)
  {
    // A fresh state per waypoint is deliberate: state spaces like ModelBasedStateSpace cache validity verdicts
    // inside the state (cleared by samplers, but not by copyFromReals), so reusing one scratch state across
    // waypoints would let a stale verdict short-circuit the validity check of the next waypoint
    ompl::base::ScopedState<> candidate_state(si_);
    Motion* motion = nullptr;

    if (goal_is_sampleable && step == seed_trajectory_.size() - 1)
    {
      // The seed endpoint only approximately matches the requested goal, so connect to a freshly sampled goal
      // state instead of the cached endpoint
      bool connected = false;
      for (int attempt = 0; attempt < cache_point_attempts_ && !connected && !ptc; ++attempt)
      {
        const ompl::base::State* goal_state = getGoalState(attempt, ptc);
        if (!goal_state)
          break;
        if (si_->equalStates(prev_motion->state, goal_state))
        {
          // The previous state already coincides with the goal state; appending it again would create a
          // zero-length edge
          motion = prev_motion;
          connected = true;
        }
        else if (si_->checkMotion(prev_motion->state, goal_state))
        {
          si_->copyState(candidate_state.get(), goal_state);
          connected = true;
        }
      }
      if (!connected)
      {
        OMPL_DEBUG("%s: Could not connect to a sampled goal state from waypoint %zu", getName().c_str(), step);
        continue;
      }
    }
    else
    {
      // The first waypoint typically equals the start state and is skipped by the duplicate check inside
      // sampleCacheStateRegion; when the actual start deviates from the seed's start, waypoint 0 is
      // reconstructed like any other waypoint
      if (!sampleCacheStateRegion(seed_trajectory_[step], candidate_state.get(), prev_motion))
      {
        // The waypoint duplicates the previous state or no collision-free state exists near it; skip it and
        // try to connect the next waypoint directly to the last reachable one
        OMPL_DEBUG("%s: Skipping seed waypoint %zu", getName().c_str(), step);
        continue;
      }
    }

    if (!motion)
    {
      motion = new Motion(si_);
      si_->copyState(motion->state, candidate_state.get());
      motion->parent = prev_motion;
      motions_.push_back(motion);
      prev_motion = motion;
    }

    double distance_to_goal = 0.0;
    if (goal->isSatisfied(motion->state, &distance_to_goal))
    {
      std::vector<Motion*> motion_path;
      for (Motion* solution = motion; solution != nullptr; solution = solution->parent)
        motion_path.push_back(solution);

      auto path(std::make_shared<ompl::geometric::PathGeometric>(si_));
      for (auto it = motion_path.rbegin(); it != motion_path.rend(); ++it)
        path->append((*it)->state);

      pdef_->addSolutionPath(path, false, distance_to_goal, getName());
      OMPL_DEBUG("%s: Reconstructed the seed trajectory with %zu states", getName().c_str(), motion_path.size());
      return true;
    }
  }
  return false;
}

ompl::base::PlannerStatus CachePlanning::solve(const ompl::base::PlannerTerminationCondition& ptc)
{
  checkValidity();

  if (!validateSeedDimensions())
    return ompl::base::PlannerStatus::ABORT;
  if (seed_trajectory_.empty())
  {
    OMPL_DEBUG("%s: No seed trajectory was set, nothing to reconstruct", getName().c_str());
    return ompl::base::PlannerStatus::ABORT;
  }

  Motion* start_motion = nullptr;
  while (const ompl::base::State* state = pis_.nextStart())
  {
    auto* motion = new Motion(si_);
    si_->copyState(motion->state, state);
    motions_.push_back(motion);
    start_motion = motion;
  }

  if (!start_motion)
  {
    // solve() may be called again without clear() in between; reuse a start motion from the existing tree
    for (Motion* motion : motions_)
    {
      if (!motion->parent)
        start_motion = motion;
    }
  }
  if (!start_motion)
  {
    OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
    return ompl::base::PlannerStatus::INVALID_START;
  }

  if (!sampler_)
    sampler_ = si_->allocStateSampler();

  bool solved = false;
  for (int attempt = 0; attempt < cache_trajectory_attempts_ && !solved && !ptc; ++attempt)
  {
    // Motions left over from a failed attempt would distort the next one, so start from the start states only
    if (attempt > 0)
      resetTreeToStartMotions();
    solved = cacheLoop(ptc, start_motion);
  }

  if (solved)
    return { true, false };
  // ABORT (not TIMEOUT) keeps the fail-fast contract visible to callers: a failed reconstruction within budget
  // must not look like an exhausted planning budget to fallback logic
  return ptc ? ompl::base::PlannerStatus(ompl::base::PlannerStatus::TIMEOUT) :
               ompl::base::PlannerStatus(ompl::base::PlannerStatus::ABORT);
}

void CachePlanning::getPlannerData(ompl::base::PlannerData& data) const
{
  Planner::getPlannerData(data);

  for (const Motion* motion : motions_)
  {
    if (motion->parent == nullptr)
    {
      data.addStartVertex(ompl::base::PlannerDataVertex(motion->state));
    }
    else
    {
      data.addEdge(ompl::base::PlannerDataVertex(motion->parent->state), ompl::base::PlannerDataVertex(motion->state));
    }
  }
}
}  // namespace ompl_interface

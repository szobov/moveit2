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

#pragma once

#include <ompl/base/Planner.h>
#include <ompl/base/StateSampler.h>

#include <algorithm>
#include <vector>

namespace ompl_interface
{
/** \brief Planner that reconstructs a previously planned "seed" trajectory in the current planning scene.
 *
 * The seed trajectory (e.g. fetched from a trajectory cache) is never replayed directly. Instead, the planner
 * re-validates it against the current scene, waypoint by waypoint: the exact waypoint is used when the motion
 * to it is valid, otherwise states within a distance of \e cache_area around it are sampled up to
 * \e cache_point_attempts times. A waypoint that cannot be connected is skipped, so the reconstructed path
 * shortcuts past it. The last waypoint is replaced by a freshly sampled goal state, so a seed whose endpoint
 * only approximately matches the requested goal still connects to the actual goal. Reconstruction of the whole
 * trajectory is retried up to \e cache_trajectory_attempts times.
 *
 * The planner only ever produces exact solutions. When the seed cannot be reconstructed (or no seed was set) it
 * fails quickly with PlannerStatus::ABORT, so the caller can fall back to a regular planner. A stale seed can
 * therefore never produce a trajectory that is invalid in the current scene.
 *
 * The seed is set with setSeedTrajectory() and is deliberately kept across clear() calls, which reset the
 * search tree only. A seed waypoint identical to the previous state (in particular a first waypoint equal to
 * the start state) is skipped. When several start states are supplied, only the most recently added one is used
 * as the reconstruction root. */
class CachePlanning : public ompl::base::Planner
{
public:
  explicit CachePlanning(const ompl::base::SpaceInformationPtr& si);

  ~CachePlanning() override;

  /** \brief Set the seed trajectory to reconstruct. Every waypoint must hold one value per value location of
   * the state space (see ompl::base::StateSpace::copyFromReals), ordered as the state space expects them.
   * The dimensions are validated in setup(); a mismatch rejects the whole seed. */
  void setSeedTrajectory(std::vector<std::vector<double>> seed);

  const std::vector<std::vector<double>>& getSeedTrajectory() const
  {
    return seed_trajectory_;
  }

  ompl::base::PlannerStatus solve(const ompl::base::PlannerTerminationCondition& ptc) override;

  /** \brief Clear the search tree. The seed trajectory and parameters are kept, so the same seed can be
   * retried in an updated scene. */
  void clear() override;

  void setup() override;

  void getPlannerData(ompl::base::PlannerData& data) const override;

  /** \brief Set the number of sampling attempts around a blocked seed waypoint (and the number of goal
   * samples tried for the final waypoint). Values below 1 are clamped to 1. */
  void setCachePointAttempts(int cache_point_attempts)
  {
    cache_point_attempts_ = std::max(1, cache_point_attempts);
  }

  int getCachePointAttempts() const
  {
    return cache_point_attempts_;
  }

  /** \brief Set the number of attempts to reconstruct the whole seed trajectory. Values below 1 are clamped
   * to 1. */
  void setCacheTrajectoryAttempts(int cache_trajectory_attempts)
  {
    cache_trajectory_attempts_ = std::max(1, cache_trajectory_attempts);
  }

  int getCacheTrajectoryAttempts() const
  {
    return cache_trajectory_attempts_;
  }

  /** \brief Set the radius of the region sampled around a blocked seed waypoint. Negative values are clamped
   * to 0 (only exact waypoints are used). */
  void setCacheArea(double cache_area)
  {
    cache_area_ = std::max(0.0, cache_area);
  }

  double getCacheArea() const
  {
    return cache_area_;
  }

protected:
  /** \brief Representation of a motion. Only parent pointers are needed, as the path is recovered by walking
   * backwards from the goal. */
  class Motion
  {
  public:
    /** \brief Constructor that allocates memory for the state */
    Motion(const ompl::base::SpaceInformationPtr& si) : state(si->allocState())
    {
    }

    ompl::base::State* state{ nullptr };
    Motion* parent{ nullptr };
  };

  /** \brief Free all motions in the search tree */
  void freeMemory();

  /** \brief Remove every motion except the start states, so a reconstruction attempt starts from a clean tree */
  void resetTreeToStartMotions();

  /** \brief Find a valid state for the given seed waypoint that can be connected to \e prev_motion: the exact
   * waypoint first, then up to \e cache_point_attempts samples within \e cache_area around it. The result is
   * written to \e state. */
  bool sampleCacheStateRegion(const std::vector<double>& waypoint, ompl::base::State* state, const Motion* prev_motion);

  /** \brief Get the goal state with the given index, fetching new goal states from the goal sampler as needed.
   * Fetched goal states are kept, so reconstruction retries can reuse them. A fetch waits (on \e ptc) while no
   * goal state is known yet or while a lazy goal sampler is still actively producing states; otherwise it does
   * not wait, so an exhausted goal sampler cannot stall the planner. Returns nullptr when no more goal states
   * can be obtained. */
  const ompl::base::State* getGoalState(std::size_t index, const ompl::base::PlannerTerminationCondition& ptc);

  /** \brief Check that every seed waypoint has one value per value location of the state space; clears the
   * seed and returns false on a mismatch. An empty seed is valid. */
  bool validateSeedDimensions();

  /** \brief Free the goal states fetched from the goal sampler */
  void freeGoalStates();

  /** \brief One attempt at reconstructing the whole seed trajectory. Returns true when a motion satisfying the
   * goal was found and the solution path was added to the problem definition. */
  bool cacheLoop(const ompl::base::PlannerTerminationCondition& ptc, Motion* start_motion);

  std::vector<std::vector<double>> seed_trajectory_;

  /** \brief Goal states fetched from the goal sampler, owned by the planner */
  std::vector<ompl::base::State*> goal_states_;

  ompl::base::StateSamplerPtr sampler_;

  /** \brief All motions created during reconstruction; start motions are the ones without a parent */
  std::vector<Motion*> motions_;

  /** \brief The number of attempts to find a collision-free state around a blocked seed waypoint */
  int cache_point_attempts_{ 5 };

  /** \brief The number of attempts to reconstruct the whole seed trajectory */
  int cache_trajectory_attempts_{ 1 };

  /** \brief The radius of the region sampled around a blocked seed waypoint */
  double cache_area_{ 0.01 };
};
}  // namespace ompl_interface

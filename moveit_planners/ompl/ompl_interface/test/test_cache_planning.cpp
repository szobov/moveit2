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

/** Unit tests for the CachePlanning seed-reconstruction planner.
 *
 * The tests run on a plain 2D RealVectorStateSpace with axis-aligned box obstacles, so they exercise the
 * planner's algorithm (exact-waypoint connection, resampling around blocked waypoints, waypoint skipping,
 * goal snapping, termination) without any robot model.
 */

#include <gtest/gtest.h>

#include <moveit/ompl_interface/detail/cache_planning.hpp>

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/goals/GoalStates.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/PathGeometric.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace ob = ompl::base;

constexpr double EPSILON = 1e-9;

class CachePlanningTest : public testing::Test
{
protected:
  void SetUp() override
  {
    auto space = std::make_shared<ob::RealVectorStateSpace>(2);
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0.0);
    bounds.setHigh(10.0);
    space->setBounds(bounds);

    si_ = std::make_shared<ob::SpaceInformation>(space);
    si_->setStateValidityChecker([this](const ob::State* state) { return isValid(state); });
    si_->setStateValidityCheckingResolution(0.0005);
    si_->setup();
  }

  bool isValid(const ob::State* state) const
  {
    const auto* real_state = state->as<ob::RealVectorStateSpace::StateType>();
    const double x = real_state->values[0];
    const double y = real_state->values[1];
    for (const std::array<double, 4>& box : obstacles_)
    {
      if (x >= box[0] && x <= box[1] && y >= box[2] && y <= box[3])
        return false;
    }
    return true;
  }

  /** \brief Create a problem definition with the given start state and a single goal state */
  ob::ProblemDefinitionPtr makeProblem(const std::array<double, 2>& start, const std::array<double, 2>& goal) const
  {
    auto pdef = std::make_shared<ob::ProblemDefinition>(si_);

    ob::ScopedState<> start_state(si_->getStateSpace());
    start_state[0] = start[0];
    start_state[1] = start[1];
    pdef->addStartState(start_state);

    auto goal_states = std::make_shared<ob::GoalStates>(si_);
    ob::ScopedState<> goal_state(si_->getStateSpace());
    goal_state[0] = goal[0];
    goal_state[1] = goal[1];
    goal_states->addState(goal_state);
    pdef->setGoal(goal_states);

    return pdef;
  }

  /** \brief A straight-line seed along y = 0, from x = 0 to x = 9 in steps of 1 */
  static std::vector<std::vector<double>> straightLineSeed()
  {
    std::vector<std::vector<double>> seed;
    for (int x = 0; x <= 9; ++x)
      seed.push_back({ static_cast<double>(x), 0.0 });
    return seed;
  }

  std::vector<double> stateToVector(const ob::State* state) const
  {
    const auto* real_state = state->as<ob::RealVectorStateSpace::StateType>();
    return { real_state->values[0], real_state->values[1] };
  }

  static std::shared_ptr<ompl::geometric::PathGeometric> solutionPath(const ompl_interface::CachePlanning& planner)
  {
    return std::dynamic_pointer_cast<ompl::geometric::PathGeometric>(planner.getProblemDefinition()->getSolutionPath());
  }

  ob::SpaceInformationPtr si_;
  std::vector<std::array<double, 4>> obstacles_;  // { x_min, x_max, y_min, y_max }
};

TEST_F(CachePlanningTest, ReconstructsValidSeedExactly)
{
  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setSeedTrajectory(straightLineSeed());
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  ASSERT_EQ(status, ob::PlannerStatus::EXACT_SOLUTION);

  auto path = solutionPath(planner);
  ASSERT_NE(path, nullptr);
  EXPECT_TRUE(path->check());

  // In an unchanged scene every waypoint connects exactly, so the solution is the seed itself
  ASSERT_EQ(path->getStateCount(), straightLineSeed().size());
  for (std::size_t i = 0; i < path->getStateCount(); ++i)
  {
    const std::vector<double> point = stateToVector(path->getState(i));
    EXPECT_NEAR(point[0], static_cast<double>(i), EPSILON);
    EXPECT_NEAR(point[1], 0.0, EPSILON);
  }
}

TEST_F(CachePlanningTest, SkipsUnreachableWaypoint)
{
  // The detour waypoint (1, 1) is blocked and cache_area is too small to escape the obstacle, so the planner
  // must skip the waypoint and connect its neighbors directly
  obstacles_.push_back({ 0.9, 1.1, 0.9, 1.1 });

  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 2.0, 0.0 }));
  planner.setSeedTrajectory({ { 0.0, 0.0 }, { 1.0, 1.0 }, { 2.0, 0.0 } });
  planner.setCacheArea(1e-6);
  planner.setCachePointAttempts(5);
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  ASSERT_EQ(status, ob::PlannerStatus::EXACT_SOLUTION);

  auto path = solutionPath(planner);
  ASSERT_NE(path, nullptr);
  EXPECT_TRUE(path->check());

  // The blocked waypoint is skipped, leaving the direct start-goal connection
  ASSERT_EQ(path->getStateCount(), 2u);
  EXPECT_NEAR(stateToVector(path->getState(1))[0], 2.0, EPSILON);
  EXPECT_NEAR(stateToVector(path->getState(1))[1], 0.0, EPSILON);
}

TEST_F(CachePlanningTest, ResamplesAroundBlockedWaypoint)
{
  // A small obstacle blocks the seed waypoint (5, 0) and the straight line through it, so the planner has to
  // find a valid state within cache_area around the waypoint
  obstacles_.push_back({ 4.99, 5.01, -0.01, 0.01 });

  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setSeedTrajectory(straightLineSeed());
  planner.setCacheArea(0.05);
  planner.setCachePointAttempts(100);
  planner.setCacheTrajectoryAttempts(5);
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  ASSERT_EQ(status, ob::PlannerStatus::EXACT_SOLUTION);

  auto path = solutionPath(planner);
  ASSERT_NE(path, nullptr);
  EXPECT_TRUE(path->check());

  // Every solution state is on the seed or near a seed waypoint. sampleUniformNear samples each dimension
  // within +/- cache_area for real vector spaces, so the Euclidean distance can reach cache_area * sqrt(2).
  const std::vector<std::vector<double>> seed = straightLineSeed();
  for (std::size_t i = 0; i < path->getStateCount(); ++i)
  {
    const std::vector<double> point = stateToVector(path->getState(i));
    double min_distance = std::numeric_limits<double>::infinity();
    for (const std::vector<double>& waypoint : seed)
      min_distance = std::min(min_distance, std::hypot(point[0] - waypoint[0], point[1] - waypoint[1]));
    EXPECT_LE(min_distance, 0.05 * std::sqrt(2.0) + EPSILON);
  }
}

TEST_F(CachePlanningTest, FailsCleanlyWithoutSeed)
{
  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  // ABORT, not TIMEOUT: fallback logic must be able to tell a fast reconstruction failure from an exhausted
  // planning budget
  EXPECT_EQ(status, ob::PlannerStatus::ABORT);
  EXPECT_EQ(planner.getProblemDefinition()->getSolutionCount(), 0u);
}

TEST_F(CachePlanningTest, RejectsSeedWithWrongDimension)
{
  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setSeedTrajectory({ { 0.0, 0.0 }, { 1.0, 0.0, 42.0 }, { 2.0, 0.0 } });
  planner.setup();

  EXPECT_TRUE(planner.getSeedTrajectory().empty());
  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  EXPECT_FALSE(status);
}

TEST_F(CachePlanningTest, FailsWhenSeedIsFullyBlocked)
{
  // A wall across the whole space separates start and goal; no reconstruction can succeed
  obstacles_.push_back({ 4.0, 6.0, 0.0, 10.0 });

  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setSeedTrajectory(straightLineSeed());
  planner.setCacheTrajectoryAttempts(3);
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  EXPECT_FALSE(status);
  EXPECT_EQ(planner.getProblemDefinition()->getSolutionCount(), 0u);
}

TEST_F(CachePlanningTest, ReturnsImmediatelyOnExpiredTerminationCondition)
{
  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setSeedTrajectory(straightLineSeed());
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::plannerAlwaysTerminatingCondition());
  EXPECT_FALSE(status);
}

TEST_F(CachePlanningTest, ClearPreservesSeedAndParameters)
{
  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setSeedTrajectory(straightLineSeed());

  std::map<std::string, std::string> params{ { "cache_area", "0.2" },
                                             { "cache_point_attempts", "7" },
                                             { "cache_trajectory_attempts", "3" } };
  planner.params().setParams(params, true);
  planner.setup();

  planner.clear();

  // clear() is called before every solve by the OMPL interface; the seed and parameters must survive it
  EXPECT_EQ(planner.getSeedTrajectory().size(), straightLineSeed().size());
  EXPECT_DOUBLE_EQ(planner.getCacheArea(), 0.2);
  EXPECT_EQ(planner.getCachePointAttempts(), 7);
  EXPECT_EQ(planner.getCacheTrajectoryAttempts(), 3);

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  EXPECT_EQ(status, ob::PlannerStatus::EXACT_SOLUTION);
}

TEST_F(CachePlanningTest, SnapsLastWaypointToSampledGoal)
{
  // The seed's endpoint (8.8, 0.2) only approximately matches the requested goal (9, 0); the solution must
  // end in the actual goal state
  std::vector<std::vector<double>> seed = straightLineSeed();
  seed.back() = { 8.8, 0.2 };

  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setSeedTrajectory(seed);
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  ASSERT_EQ(status, ob::PlannerStatus::EXACT_SOLUTION);

  auto path = solutionPath(planner);
  ASSERT_NE(path, nullptr);
  const std::vector<double> last = stateToVector(path->getState(path->getStateCount() - 1));
  EXPECT_NEAR(last[0], 9.0, EPSILON);
  EXPECT_NEAR(last[1], 0.0, EPSILON);
}

TEST_F(CachePlanningTest, SingleWaypointSeedConnectsStartToGoal)
{
  // A one-waypoint seed (e.g. a truncated reference trajectory) must still attempt the direct start-goal
  // connection through the goal-snapping branch
  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setSeedTrajectory({ { 9.0, 0.0 } });
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  ASSERT_EQ(status, ob::PlannerStatus::EXACT_SOLUTION);

  auto path = solutionPath(planner);
  ASSERT_NE(path, nullptr);
  ASSERT_EQ(path->getStateCount(), 2u);
  EXPECT_NEAR(stateToVector(path->getState(1))[0], 9.0, EPSILON);
  EXPECT_NEAR(stateToVector(path->getState(1))[1], 0.0, EPSILON);
}

TEST_F(CachePlanningTest, ReconstructsWhenStartDeviatesFromSeed)
{
  // The seed starts at (0, 2) but the robot is at (0, 0), with a wall that blocks the direct connection to the
  // seed's second waypoint. The seed's first waypoint must be reconstructed like any other waypoint, not
  // silently discarded.
  obstacles_.push_back({ 1.0, 3.0, 0.0, 1.5 });

  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 4.0, 2.0 }));
  planner.setSeedTrajectory({ { 0.0, 2.0 }, { 2.0, 2.0 }, { 4.0, 2.0 } });
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  ASSERT_EQ(status, ob::PlannerStatus::EXACT_SOLUTION);

  auto path = solutionPath(planner);
  ASSERT_NE(path, nullptr);
  EXPECT_TRUE(path->check());
  // start, the seed's first two waypoints, and the goal
  EXPECT_EQ(path->getStateCount(), 4u);
}

TEST_F(CachePlanningTest, HandlesGoalEqualToStart)
{
  // When the start state already satisfies the goal, the sampled goal state coincides with the previous state;
  // the solution must not contain a zero-length edge
  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 5.0, 5.0 }, { 5.0, 5.0 }));
  planner.setSeedTrajectory({ { 5.0, 5.0 }, { 5.0, 5.0 } });
  planner.setup();

  ob::PlannerStatus status = planner.solve(ob::timedPlannerTerminationCondition(5.0));
  ASSERT_EQ(status, ob::PlannerStatus::EXACT_SOLUTION);

  auto path = solutionPath(planner);
  ASSERT_NE(path, nullptr);
  EXPECT_EQ(path->getStateCount(), 1u);
}

TEST_F(CachePlanningTest, SecondSolveWithoutClearSucceeds)
{
  // The OMPL Planner contract allows repeated solve() calls without clear() in between
  ompl_interface::CachePlanning planner(si_);
  planner.setProblemDefinition(makeProblem({ 0.0, 0.0 }, { 9.0, 0.0 }));
  planner.setSeedTrajectory(straightLineSeed());
  planner.setup();

  EXPECT_EQ(planner.solve(ob::timedPlannerTerminationCondition(5.0)), ob::PlannerStatus::EXACT_SOLUTION);
  EXPECT_EQ(planner.solve(ob::timedPlannerTerminationCondition(5.0)), ob::PlannerStatus::EXACT_SOLUTION);
}

TEST_F(CachePlanningTest, ClampsInvalidParameters)
{
  ompl_interface::CachePlanning planner(si_);
  planner.setCachePointAttempts(0);
  planner.setCacheTrajectoryAttempts(-5);
  planner.setCacheArea(-0.1);

  EXPECT_EQ(planner.getCachePointAttempts(), 1);
  EXPECT_EQ(planner.getCacheTrajectoryAttempts(), 1);
  EXPECT_DOUBLE_EQ(planner.getCacheArea(), 0.0);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

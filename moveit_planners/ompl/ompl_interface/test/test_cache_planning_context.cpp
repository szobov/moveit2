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

/** Tests for seeding CachePlanning through MotionPlanRequest.reference_trajectories.
 *
 * The tests build ModelBasedPlanningContexts through the PlanningContextManager, exactly as the OMPL planner
 * manager does, and verify that:
 *  - a reference trajectory in the request reaches a CachePlanning planner as its seed, reordered to the state
 *    space's variable order;
 *  - invalid reference trajectories (missing joints, inconsistent points) are dropped so the planner fails
 *    cleanly instead of reconstructing a wrong trajectory;
 *  - the full flow works: a trajectory planned by RRTConnect seeds CachePlanning, which reconstructs it in the
 *    current scene, skips waypoints that became invalid, and fails on a missing seed so a caller can fall back
 *    to a regular planner.
 */

#include "load_test_robot.hpp"

#include <gtest/gtest.h>

#include <moveit/constraint_samplers/constraint_sampler_manager.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/ompl_interface/detail/cache_planning.hpp>
#include <moveit/ompl_interface/planning_context_manager.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/utils/logger.hpp>

#include <geometric_shapes/shapes.h>
#include <moveit_msgs/msg/generic_trajectory.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace
{
constexpr double GOAL_TOLERANCE = 0.001;

/** \brief The panda 'ready' state, known to be within limits and self-collision free */
const std::vector<double> START_STATE = { 0., -0.785, 0., -2.356, 0., 1.571, 0.785 };
/** \brief The 'ready' state with the last joint moved, so start and goal connect on a straight line */
const std::vector<double> GOAL_STATE = { 0., -0.785, 0., -2.356, 0., 1.571, 0.685 };
}  // namespace

class TestCachePlanningContext : public ompl_interface_testing::LoadTestRobot, public testing::Test
{
public:
  TestCachePlanningContext() : TestCachePlanningContext("panda", "panda_arm")
  {
  }

protected:
  TestCachePlanningContext(const std::string& robot_name, const std::string& group_name)
    : LoadTestRobot(robot_name, group_name), node_(std::make_shared<rclcpp::Node>("cache_planning_context_test"))
  {
    moveit::setNodeLoggerName(node_->get_name());
  }

  void SetUp() override
  {
    constraint_sampler_manager_ = std::make_shared<constraint_samplers::ConstraintSamplerManager>();
    planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_);
  }

  /** \brief Create a planning request to plan from a given start state to a joint space goal */
  planning_interface::MotionPlanRequest createRequest(const std::vector<double>& start,
                                                      const std::vector<double>& goal) const
  {
    planning_interface::MotionPlanRequest request;
    request.group_name = group_name_;
    request.allowed_planning_time = 5.0;

    moveit::core::RobotState start_state(robot_model_);
    start_state.setToDefaultValues();
    start_state.setJointGroupPositions(joint_model_group_, start);
    moveit::core::robotStateToRobotStateMsg(start_state, request.start_state);

    moveit::core::RobotState goal_state(robot_model_);
    goal_state.setToDefaultValues();
    goal_state.setJointGroupPositions(joint_model_group_, goal);
    request.goal_constraints.push_back(
        kinematic_constraints::constructGoalConstraints(goal_state, joint_model_group_, GOAL_TOLERANCE));

    return request;
  }

  /** \brief Turn joint waypoints into the request's reference trajectories */
  void setReferenceTrajectory(planning_interface::MotionPlanRequest& request,
                              const std::vector<std::vector<double>>& waypoints,
                              const std::vector<std::string>& joint_names) const
  {
    moveit_msgs::msg::GenericTrajectory reference_trajectory;
    trajectory_msgs::msg::JointTrajectory joint_trajectory;
    joint_trajectory.joint_names = joint_names;
    for (const std::vector<double>& waypoint : waypoints)
    {
      trajectory_msgs::msg::JointTrajectoryPoint point;
      point.positions = waypoint;
      joint_trajectory.points.push_back(point);
    }
    reference_trajectory.joint_trajectory.push_back(joint_trajectory);
    request.reference_trajectories = { reference_trajectory };
  }

  /** \brief Linear joint-space interpolation from start to goal with the given number of waypoints */
  std::vector<std::vector<double>> interpolateWaypoints(const std::vector<double>& start,
                                                        const std::vector<double>& goal,
                                                        std::size_t num_waypoints) const
  {
    std::vector<std::vector<double>> waypoints;
    waypoints.reserve(num_waypoints);
    for (std::size_t i = 0; i < num_waypoints; ++i)
    {
      const double fraction = static_cast<double>(i) / static_cast<double>(num_waypoints - 1);
      std::vector<double> waypoint(start.size());
      joint_model_group_->interpolate(start.data(), goal.data(), fraction, waypoint.data());
      waypoints.push_back(std::move(waypoint));
    }
    return waypoints;
  }

  /** \brief Get a planning context whose planner is selected by the given configuration */
  ompl_interface::ModelBasedPlanningContextPtr
  createContext(const planning_interface::MotionPlanRequest& request,
                const std::map<std::string, std::string>& planner_config) const
  {
    planning_interface::PlannerConfigurationSettings pconfig_settings;
    pconfig_settings.group = group_name_;
    pconfig_settings.name = group_name_;
    pconfig_settings.config = planner_config;

    ompl_interface::PlanningContextManager pcm(robot_model_, constraint_sampler_manager_);
    pcm.setPlannerConfigurations({ { pconfig_settings.name, pconfig_settings } });

    moveit_msgs::msg::MoveItErrorCodes error_code;
    return pcm.getPlanningContext(planning_scene_, request, error_code, node_, false);
  }

  /** \brief Planner configuration for CachePlanning with post-processing disabled, so the solution trajectory
   * can be compared against the seed waypoint by waypoint */
  static std::map<std::string, std::string> cachePlanningConfig()
  {
    return { { "type", "geometric::CachePlanning" },
             { "enforce_joint_model_state_space", "1" },
             { "simplify_solutions", "0" },
             { "interpolate", "0" } };
  }

  static std::map<std::string, std::string> rrtConnectConfig()
  {
    return { { "type", "geometric::RRTConnect" }, { "enforce_joint_model_state_space", "1" } };
  }

  const ompl_interface::CachePlanning* getCachePlanner(const ompl_interface::ModelBasedPlanningContextPtr& context)
  {
    return dynamic_cast<const ompl_interface::CachePlanning*>(context->getOMPLSimpleSetup()->getPlanner().get());
  }

  /** \brief Assert that the planner's seed equals the given waypoints, value by value */
  void expectSeedEquals(const ompl_interface::CachePlanning* planner,
                        const std::vector<std::vector<double>>& waypoints) const
  {
    ASSERT_EQ(planner->getSeedTrajectory().size(), waypoints.size());
    for (std::size_t i = 0; i < waypoints.size(); ++i)
    {
      ASSERT_EQ(planner->getSeedTrajectory()[i].size(), num_dofs_);
      for (std::size_t j = 0; j < num_dofs_; ++j)
        EXPECT_DOUBLE_EQ(planner->getSeedTrajectory()[i][j], waypoints[i][j]);
    }
  }

  rclcpp::Node::SharedPtr node_;
  planning_scene::PlanningScenePtr planning_scene_;
  constraint_samplers::ConstraintSamplerManagerPtr constraint_sampler_manager_;
};

TEST_F(TestCachePlanningContext, ReferenceTrajectoryReachesPlannerAsSeed)
{
  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);
  const std::vector<std::vector<double>> waypoints = interpolateWaypoints(START_STATE, GOAL_STATE, 5);
  setReferenceTrajectory(request, waypoints, joint_model_group_->getVariableNames());

  auto context = createContext(request, cachePlanningConfig());
  ASSERT_NE(context, nullptr);

  const ompl_interface::CachePlanning* planner = getCachePlanner(context);
  ASSERT_NE(planner, nullptr) << "The configured planner is not CachePlanning";
  expectSeedEquals(planner, waypoints);
}

TEST_F(TestCachePlanningContext, ShuffledJointNamesAreReordered)
{
  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);
  const std::vector<std::vector<double>> waypoints = interpolateWaypoints(START_STATE, GOAL_STATE, 5);

  // Reverse the joint order in the reference trajectory; the seed must come out in state space order anyway
  std::vector<std::string> reversed_names = joint_model_group_->getVariableNames();
  std::reverse(reversed_names.begin(), reversed_names.end());
  std::vector<std::vector<double>> reversed_waypoints = waypoints;
  for (std::vector<double>& waypoint : reversed_waypoints)
    std::reverse(waypoint.begin(), waypoint.end());
  setReferenceTrajectory(request, reversed_waypoints, reversed_names);

  auto context = createContext(request, cachePlanningConfig());
  ASSERT_NE(context, nullptr);

  const ompl_interface::CachePlanning* planner = getCachePlanner(context);
  ASSERT_NE(planner, nullptr);
  expectSeedEquals(planner, waypoints);
}

TEST_F(TestCachePlanningContext, MissingJointDropsSeed)
{
  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);
  std::vector<std::string> incomplete_names = joint_model_group_->getVariableNames();
  incomplete_names.pop_back();
  std::vector<std::vector<double>> incomplete_waypoints = interpolateWaypoints(START_STATE, GOAL_STATE, 5);
  for (std::vector<double>& waypoint : incomplete_waypoints)
    waypoint.pop_back();
  setReferenceTrajectory(request, incomplete_waypoints, incomplete_names);

  auto context = createContext(request, cachePlanningConfig());
  ASSERT_NE(context, nullptr);

  const ompl_interface::CachePlanning* planner = getCachePlanner(context);
  ASSERT_NE(planner, nullptr);
  EXPECT_TRUE(planner->getSeedTrajectory().empty());
}

TEST_F(TestCachePlanningContext, InconsistentTrajectoryPointDropsSeed)
{
  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);
  std::vector<std::vector<double>> waypoints = interpolateWaypoints(START_STATE, GOAL_STATE, 5);
  waypoints[2].push_back(0.42);
  setReferenceTrajectory(request, waypoints, joint_model_group_->getVariableNames());

  auto context = createContext(request, cachePlanningConfig());
  ASSERT_NE(context, nullptr);

  const ompl_interface::CachePlanning* planner = getCachePlanner(context);
  ASSERT_NE(planner, nullptr);
  EXPECT_TRUE(planner->getSeedTrajectory().empty());
}

TEST_F(TestCachePlanningContext, ReconstructsSeededTrajectory)
{
  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);
  const std::vector<std::vector<double>> waypoints = interpolateWaypoints(START_STATE, GOAL_STATE, 5);
  setReferenceTrajectory(request, waypoints, joint_model_group_->getVariableNames());

  auto context = createContext(request, cachePlanningConfig());
  ASSERT_NE(context, nullptr);

  planning_interface::MotionPlanResponse response;
  context->solve(response);
  ASSERT_EQ(response.error_code.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  ASSERT_NE(response.trajectory, nullptr);
  EXPECT_TRUE(planning_scene_->isPathValid(*response.trajectory, group_name_));

  // In an unchanged scene every waypoint connects exactly, so the solution follows the seed; the last waypoint
  // is a sampled goal state within the goal tolerance
  ASSERT_EQ(response.trajectory->getWayPointCount(), waypoints.size());
  for (std::size_t i = 0; i < waypoints.size(); ++i)
  {
    std::vector<double> solution_waypoint;
    response.trajectory->getWayPoint(i).copyJointGroupPositions(joint_model_group_, solution_waypoint);
    for (std::size_t j = 0; j < num_dofs_; ++j)
    {
      const double tolerance = (i == waypoints.size() - 1) ? GOAL_TOLERANCE : 1e-9;
      EXPECT_NEAR(solution_waypoint[j], waypoints[i][j], tolerance);
    }
  }
}

TEST_F(TestCachePlanningContext, SkipsWaypointInvalidatedByObstacle)
{
  // The seed makes a detour: the middle waypoint swings the arm to the side. An obstacle placed at that
  // waypoint's end-effector position invalidates it, while the direct start-goal motion stays far away from
  // the obstacle. The planner must skip the blocked waypoint and connect start to goal directly.
  std::vector<double> detour_state_values = START_STATE;
  detour_state_values[0] += 1.5;

  moveit::core::RobotState detour_state(robot_model_);
  detour_state.setToDefaultValues();
  detour_state.setJointGroupPositions(joint_model_group_, detour_state_values);
  detour_state.update();
  const Eigen::Isometry3d obstacle_pose = detour_state.getGlobalLinkTransform(ee_link_name_);

  planning_scene_->getWorldNonConst()->addToObject("obstacle", std::make_shared<shapes::Box>(0.15, 0.15, 0.15),
                                                   obstacle_pose);
  ASSERT_TRUE(planning_scene_->isStateColliding(detour_state, group_name_))
      << "The obstacle must invalidate the detour waypoint for this test to be meaningful";

  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);
  setReferenceTrajectory(request, { START_STATE, detour_state_values, GOAL_STATE },
                         joint_model_group_->getVariableNames());

  auto context = createContext(request, cachePlanningConfig());
  ASSERT_NE(context, nullptr);

  planning_interface::MotionPlanResponse response;
  context->solve(response);
  ASSERT_EQ(response.error_code.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  ASSERT_NE(response.trajectory, nullptr);
  EXPECT_TRUE(planning_scene_->isPathValid(*response.trajectory, group_name_));

  // The blocked detour waypoint is skipped, leaving the direct start-goal connection
  EXPECT_EQ(response.trajectory->getWayPointCount(), 2u);
}

TEST_F(TestCachePlanningContext, FailsWithoutSeedSoCallerCanFallBack)
{
  // A request without reference trajectories: CachePlanning must fail quickly ...
  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);

  auto cache_context = createContext(request, cachePlanningConfig());
  ASSERT_NE(cache_context, nullptr);

  planning_interface::MotionPlanResponse cache_response;
  cache_context->solve(cache_response);
  EXPECT_NE(cache_response.error_code.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);

  // ... so that the caller can fall through to a regular planner with the same request
  auto fallback_context = createContext(request, rrtConnectConfig());
  ASSERT_NE(fallback_context, nullptr);

  planning_interface::MotionPlanResponse fallback_response;
  fallback_context->solve(fallback_response);
  ASSERT_EQ(fallback_response.error_code.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
}

TEST_F(TestCachePlanningContext, PlannedTrajectorySeedsReplanning)
{
  // The full round trip: plan with a regular planner, use the result as the reference trajectory of the next
  // request, and let CachePlanning reconstruct it
  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);

  auto plan_context = createContext(request, rrtConnectConfig());
  ASSERT_NE(plan_context, nullptr);

  planning_interface::MotionPlanResponse plan_response;
  plan_context->solve(plan_response);
  ASSERT_EQ(plan_response.error_code.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  ASSERT_NE(plan_response.trajectory, nullptr);

  moveit_msgs::msg::RobotTrajectory trajectory_msg;
  plan_response.trajectory->getRobotTrajectoryMsg(trajectory_msg);
  ASSERT_FALSE(trajectory_msg.joint_trajectory.points.empty());

  moveit_msgs::msg::GenericTrajectory reference_trajectory;
  reference_trajectory.joint_trajectory.push_back(trajectory_msg.joint_trajectory);
  request.reference_trajectories = { reference_trajectory };

  auto cache_context = createContext(request, cachePlanningConfig());
  ASSERT_NE(cache_context, nullptr);

  const ompl_interface::CachePlanning* planner = getCachePlanner(cache_context);
  ASSERT_NE(planner, nullptr);
  EXPECT_EQ(planner->getSeedTrajectory().size(), trajectory_msg.joint_trajectory.points.size());

  planning_interface::MotionPlanResponse cache_response;
  cache_context->solve(cache_response);
  ASSERT_EQ(cache_response.error_code.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  ASSERT_NE(cache_response.trajectory, nullptr);
  EXPECT_TRUE(planning_scene_->isPathValid(*cache_response.trajectory, group_name_));
}

TEST_F(TestCachePlanningContext, IgnoresColumnsForOtherJoints)
{
  // External tooling often exports all URDF joints; columns that do not belong to the group's active joints
  // (fixed joints, other groups' joints) must be ignored, not rejected or worse
  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);
  const std::vector<std::vector<double>> waypoints = interpolateWaypoints(START_STATE, GOAL_STATE, 5);

  std::vector<std::string> padded_names = joint_model_group_->getVariableNames();
  padded_names.push_back("panda_joint8");  // a fixed joint the robot model knows
  std::vector<std::vector<double>> padded_waypoints = waypoints;
  for (std::vector<double>& waypoint : padded_waypoints)
    waypoint.push_back(0.0);
  setReferenceTrajectory(request, padded_waypoints, padded_names);

  auto context = createContext(request, cachePlanningConfig());
  ASSERT_NE(context, nullptr);

  const ompl_interface::CachePlanning* planner = getCachePlanner(context);
  ASSERT_NE(planner, nullptr);
  expectSeedEquals(planner, waypoints);
}

TEST_F(TestCachePlanningContext, IgnoresMalformedVelocities)
{
  // Only positions are consumed; velocity/acceleration/effort arrays of the wrong size must not be touched
  // (reading them per joint name would index out of bounds)
  planning_interface::MotionPlanRequest request = createRequest(START_STATE, GOAL_STATE);
  const std::vector<std::vector<double>> waypoints = interpolateWaypoints(START_STATE, GOAL_STATE, 5);
  setReferenceTrajectory(request, waypoints, joint_model_group_->getVariableNames());
  for (trajectory_msgs::msg::JointTrajectoryPoint& point : request.reference_trajectories[0].joint_trajectory[0].points)
  {
    point.velocities = { 0.1, 0.2 };  // wrong size on purpose
  }

  auto context = createContext(request, cachePlanningConfig());
  ASSERT_NE(context, nullptr);

  const ompl_interface::CachePlanning* planner = getCachePlanner(context);
  ASSERT_NE(planner, nullptr);
  expectSeedEquals(planner, waypoints);
}

/***************************************************************************
 * The panda 'hand' group contains a mimic joint (panda_finger_joint2 mimics panda_finger_joint1), so it
 * exercises the seed conversion for groups whose trajectories name only the active joints
 * ************************************************************************/
class TestCachePlanningContextHandGroup : public TestCachePlanningContext
{
protected:
  TestCachePlanningContextHandGroup() : TestCachePlanningContext("panda", "hand")
  {
  }
};

TEST_F(TestCachePlanningContextHandGroup, MimicJointGroupSeedRoundTrip)
{
  // group variables: panda_finger_joint1 (active) and panda_finger_joint2 (mimic)
  const std::vector<double> start = { 0.0, 0.0 };
  const std::vector<double> goal = { 0.03, 0.03 };

  planning_interface::MotionPlanRequest request;
  request.group_name = group_name_;
  request.allowed_planning_time = 5.0;

  moveit::core::RobotState start_state(robot_model_);
  start_state.setToDefaultValues();
  // the arm's all-zero default configuration is self-colliding; park it at the 'ready' pose
  start_state.setJointGroupPositions(robot_model_->getJointModelGroup("panda_arm"), START_STATE);
  start_state.setJointGroupPositions(joint_model_group_, start);
  moveit::core::robotStateToRobotStateMsg(start_state, request.start_state);

  // constructGoalConstraints() constrains every group variable, but a constraint on a mimic joint cannot be
  // sampled; constrain the active joint only
  moveit_msgs::msg::Constraints goal_constraints;
  goal_constraints.joint_constraints.resize(1);
  goal_constraints.joint_constraints[0].joint_name = "panda_finger_joint1";
  goal_constraints.joint_constraints[0].position = goal[0];
  goal_constraints.joint_constraints[0].tolerance_above = GOAL_TOLERANCE;
  goal_constraints.joint_constraints[0].tolerance_below = GOAL_TOLERANCE;
  goal_constraints.joint_constraints[0].weight = 1.0;
  request.goal_constraints = { goal_constraints };

  auto plan_context = createContext(request, rrtConnectConfig());
  ASSERT_NE(plan_context, nullptr);

  planning_interface::MotionPlanResponse plan_response;
  plan_context->solve(plan_response);
  ASSERT_EQ(plan_response.error_code.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  ASSERT_NE(plan_response.trajectory, nullptr);

  moveit_msgs::msg::RobotTrajectory trajectory_msg;
  plan_response.trajectory->getRobotTrajectoryMsg(trajectory_msg);
  // MoveIt's trajectory export names active joints only; the seed conversion must accept that
  ASSERT_EQ(trajectory_msg.joint_trajectory.joint_names, std::vector<std::string>{ "panda_finger_joint1" });

  moveit_msgs::msg::GenericTrajectory reference_trajectory;
  reference_trajectory.joint_trajectory.push_back(trajectory_msg.joint_trajectory);
  request.reference_trajectories = { reference_trajectory };

  auto cache_context = createContext(request, cachePlanningConfig());
  ASSERT_NE(cache_context, nullptr);

  const ompl_interface::CachePlanning* planner = getCachePlanner(cache_context);
  ASSERT_NE(planner, nullptr);
  ASSERT_FALSE(planner->getSeedTrajectory().empty());
  // seed rows carry the full group variable count, mimic variable included
  EXPECT_EQ(planner->getSeedTrajectory().front().size(), joint_model_group_->getVariableCount());

  planning_interface::MotionPlanResponse cache_response;
  cache_context->solve(cache_response);
  ASSERT_EQ(cache_response.error_code.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  ASSERT_NE(cache_response.trajectory, nullptr);
  EXPECT_TRUE(planning_scene_->isPathValid(*cache_response.trajectory, group_name_));
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);

  const int ret = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return ret;
}

/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, MoveIt Contributors
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
 *   * Neither the name of the copyright holder nor the names of its
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

/* Tests that a URDF collision mesh referencing a multi-object OBJ file (e.g. an offline V-HACD
   convex decomposition) yields one collision shape per object in the LinkModel, with matching
   collision origin transforms, and that every piece reaches the collision backend. */

#include <geometric_shapes/shapes.h>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection_fcl/collision_env_fcl.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace
{
std::string resourceUri(const std::string& filename)
{
  return "file://" + std::string(TEST_RESOURCES_DIR) + "/" + filename;
}

/// Build a single-link robot model whose only link has one collision element with the given geometry
moveit::core::RobotModelPtr buildModelWithCollisionGeometry(const std::string& geometry_xml,
                                                            const std::string& origin_xml)
{
  const std::string urdf_string = R"(<?xml version="1.0"?>
    <robot name="obj_collision_test">
      <link name="base_link">
        <collision>
          )" + origin_xml +
                                  R"(
          <geometry>
            )" + geometry_xml +
                                  R"(
          </geometry>
        </collision>
      </link>
    </robot>)";
  const urdf::ModelInterfaceSharedPtr urdf_model = urdf::parseURDF(urdf_string);
  if (!urdf_model)
    return nullptr;
  auto srdf_model = std::make_shared<srdf::Model>();
  srdf_model->initString(*urdf_model, R"(<robot name="obj_collision_test"/>)");
  return std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
}

Eigen::AlignedBox3d computeAabb(const shapes::Mesh& mesh)
{
  Eigen::AlignedBox3d aabb;
  for (unsigned int i = 0; i < mesh.vertex_count; ++i)
    aabb.extend(Eigen::Vector3d(mesh.vertices[3 * i], mesh.vertices[3 * i + 1], mesh.vertices[3 * i + 2]));
  return aabb;
}

/// The shapes of a link as meshes, sorted by the smallest x coordinate of their vertices.
/// Fails the current test and returns an empty vector if any shape is not a mesh.
std::vector<const shapes::Mesh*> meshesSortedByMinX(const moveit::core::LinkModel& link)
{
  std::vector<const shapes::Mesh*> meshes;
  for (const shapes::ShapeConstPtr& shape : link.getShapes())
  {
    const auto* mesh = dynamic_cast<const shapes::Mesh*>(shape.get());
    if (mesh == nullptr)
    {
      ADD_FAILURE() << "link shape is not a shapes::Mesh";
      return {};
    }
    meshes.push_back(mesh);
  }
  std::sort(meshes.begin(), meshes.end(), [](const shapes::Mesh* a, const shapes::Mesh* b) {
    return computeAabb(*a).min().x() < computeAabb(*b).min().x();
  });
  return meshes;
}
}  // namespace

TEST(MultiMeshCollision, TwoHullObjCreatesTwoShapesWithMatchingOrigins)
{
  moveit::core::RobotModelPtr model =
      buildModelWithCollisionGeometry("<mesh filename=\"" + resourceUri("two_hulls.obj") + "\"/>",
                                      R"(<origin xyz="0.1 0.2 0.3" rpy="0 0 1.5707963267948966"/>)");
  ASSERT_TRUE(model);
  const moveit::core::LinkModel* link = model->getLinkModel("base_link");
  ASSERT_TRUE(link);

  ASSERT_EQ(2u, link->getShapes().size());
  ASSERT_EQ(2u, link->getCollisionOriginTransforms().size());

  const Eigen::Isometry3d expected_origin =
      Eigen::Translation3d(0.1, 0.2, 0.3) * Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ());
  for (std::size_t i = 0; i < link->getShapes().size(); ++i)
  {
    EXPECT_EQ(shapes::MESH, link->getShapes()[i]->type) << "shape " << i;
    EXPECT_TRUE(link->getCollisionOriginTransforms()[i].isApprox(expected_origin, 1e-9)) << "origin " << i;
  }

  // Both hulls must stay disjoint pieces: one tetrahedron in x [0, 1], the other in x [5, 6]
  const std::vector<const shapes::Mesh*> meshes = meshesSortedByMinX(*link);
  ASSERT_EQ(2u, meshes.size());
  EXPECT_EQ(4u, meshes[0]->vertex_count);
  EXPECT_EQ(4u, meshes[0]->triangle_count);
  EXPECT_EQ(4u, meshes[1]->vertex_count);
  EXPECT_EQ(4u, meshes[1]->triangle_count);
  EXPECT_NEAR(1.0, computeAabb(*meshes[0]).max().x(), 1e-6);
  EXPECT_NEAR(5.0, computeAabb(*meshes[1]).min().x(), 1e-6);
}

TEST(MultiMeshCollision, MeshScaleIsAppliedExactlyOnce)
{
  moveit::core::RobotModelPtr model =
      buildModelWithCollisionGeometry("<mesh filename=\"" + resourceUri("two_hulls.obj") + "\" scale=\"2 3 4\"/>",
                                      R"(<origin xyz="0 0 0" rpy="0 0 0"/>)");
  ASSERT_TRUE(model);
  const moveit::core::LinkModel* link = model->getLinkModel("base_link");
  ASSERT_TRUE(link);
  ASSERT_EQ(2u, link->getShapes().size());

  const std::vector<const shapes::Mesh*> meshes = meshesSortedByMinX(*link);
  ASSERT_EQ(2u, meshes.size());
  // Unscaled hulls span x [0, 1] and [5, 6], y [0, 1], z [0, 1]. Scale (2, 3, 4) applied once:
  EXPECT_TRUE(computeAabb(*meshes[0]).max().isApprox(Eigen::Vector3d(2.0, 3.0, 4.0), 1e-6));
  EXPECT_NEAR(10.0, computeAabb(*meshes[1]).min().x(), 1e-6);
  EXPECT_NEAR(12.0, computeAabb(*meshes[1]).max().x(), 1e-6);
}

TEST(MultiMeshCollision, SingleObjectObjCreatesOneShape)
{
  moveit::core::RobotModelPtr model = buildModelWithCollisionGeometry(
      "<mesh filename=\"" + resourceUri("single_hull.obj") + "\"/>", R"(<origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>)");
  ASSERT_TRUE(model);
  const moveit::core::LinkModel* link = model->getLinkModel("base_link");
  ASSERT_TRUE(link);

  ASSERT_EQ(1u, link->getShapes().size());
  ASSERT_EQ(1u, link->getCollisionOriginTransforms().size());
  EXPECT_EQ(shapes::MESH, link->getShapes()[0]->type);
  EXPECT_TRUE(link->getCollisionOriginTransforms()[0].translation().isApprox(Eigen::Vector3d(0.1, 0.2, 0.3), 1e-9));
}

TEST(MultiMeshCollision, PrimitiveCollisionGeometryIsUnchanged)
{
  moveit::core::RobotModelPtr model =
      buildModelWithCollisionGeometry(R"(<box size="1 2 3"/>)", R"(<origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>)");
  ASSERT_TRUE(model);
  const moveit::core::LinkModel* link = model->getLinkModel("base_link");
  ASSERT_TRUE(link);

  ASSERT_EQ(1u, link->getShapes().size());
  ASSERT_EQ(1u, link->getCollisionOriginTransforms().size());
  ASSERT_EQ(shapes::BOX, link->getShapes()[0]->type);
  const auto& box = static_cast<const shapes::Box&>(*link->getShapes()[0]);
  EXPECT_DOUBLE_EQ(1.0, box.size[0]);
  EXPECT_DOUBLE_EQ(2.0, box.size[1]);
  EXPECT_DOUBLE_EQ(3.0, box.size[2]);
  EXPECT_TRUE(link->getCollisionOriginTransforms()[0].translation().isApprox(Eigen::Vector3d(0.1, 0.2, 0.3), 1e-9));
}

/* Place a small sphere overlapping the surface of each hull (and in the empty gap between them)
   and check world-robot collision, proving that both pieces of the decomposition reach the
   collision backend. The spheres are centered on hull vertices because FCL treats meshes as
   triangle surfaces: an object fully inside a mesh without touching any triangle is no contact. */
TEST(MultiMeshCollision, BothHullsAreDetectedByCollisionBackend)
{
  moveit::core::RobotModelPtr model = buildModelWithCollisionGeometry(
      "<mesh filename=\"" + resourceUri("two_hulls.obj") + "\"/>", R"(<origin xyz="0 0 0" rpy="0 0 0"/>)");
  ASSERT_TRUE(model);

  collision_detection::CollisionEnvFCL env(model);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  state.update();

  const auto sphere = std::make_shared<const shapes::Sphere>(0.1);
  const auto check_sphere_collision = [&](const Eigen::Vector3d& position) {
    env.getWorld()->clearObjects();
    env.getWorld()->addToObject("probe", sphere, Eigen::Isometry3d(Eigen::Translation3d(position)));
    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;
    env.checkRobotCollision(req, res, state);
    return res.collision;
  };

  EXPECT_TRUE(check_sphere_collision(Eigen::Vector3d(0.0, 0.0, 0.0))) << "sphere touching hull_0 must collide";
  EXPECT_TRUE(check_sphere_collision(Eigen::Vector3d(5.0, 0.0, 0.0))) << "sphere touching hull_1 must collide";
  EXPECT_FALSE(check_sphere_collision(Eigen::Vector3d(3.0, 0.5, 0.5))) << "sphere between the hulls must not collide";
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

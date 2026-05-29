import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer
    params = gtsam.LevenbergMarquardtParams()

    # TODO: Perform the optimization and print the result
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    return optimizer.optimize()

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None
    best_landmark = None
    min_uncertainty = float('inf')
    final_sum = 0

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            curr_graph = gtsam.NonlinearFactorGraph(graph)
            curr_estimate = gtsam.Values(initial_estimate)

            curr_graph, curr_estimate = add_pose(curr_graph, curr_estimate, pose_5)
            result1 = optimize(curr_graph, curr_estimate)

            curr_graph = add_landmark_measurement(curr_graph, result1, pose_5, landmark)
            result2 = optimize(curr_graph, result1)

            marginals = gtsam.Marginals(curr_graph, result2)
            cov1 = marginals.marginalCovariance(L(1))
            cov2 = marginals.marginalCovariance(L(2))

            uncertainty = np.trace(cov1) + np.trace(cov2)

            if uncertainty < min_uncertainty:
                min_uncertainty = uncertainty
                best_pose = pose_key
                best_landmark = landmark

                final_sum = cov1.sum() + cov2.sum()

    return best_pose, best_landmark, final_sum

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None
    best_landmark = None
    min_sum_errors = float('inf')

    true_poses = {
        1: gtsam.Pose2(0.0, 0.0, 0.0),
        2: gtsam.Pose2(2.0, 0.0, 0.0),
        3: gtsam.Pose2(4.0, 0.0, 0.0)
    }

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            curr_graph = gtsam.NonlinearFactorGraph(graph)
            curr_estimate = gtsam.Values(initial_estimate)

            curr_graph, curr_estimate = add_pose(curr_graph, curr_estimate, pose_5)
            result1 = optimize(curr_graph, curr_estimate)

            curr_graph = add_landmark_measurement(curr_graph, result1, pose_5, landmark)
            result2 = optimize(curr_graph, result1)

        # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
            list_of_errors = []
        
        # TODO: compute the sum of the errors and return it along with the best pose and landmark
            for i in [1, 2, 3]:
                est_pose = result2.atPose2(X(i))
                true_pose = true_poses[i]

                error = np.sqrt(
                    (est_pose.x() - true_pose.x())**2 + 
                    (est_pose.y() - true_pose.y())**2 + 
                    (est_pose.theta() - true_pose.theta())**2
                )
                list_of_errors.append(error)

            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < min_sum_errors:
                min_sum_errors = sum_of_errors
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, min_sum_errors
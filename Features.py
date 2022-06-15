import numpy as np
import pandas as pd
import os
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt

def one_hots(labels=''):
    new_word_id = 0
    words = []
    dictionary = {}
    labels = labels[:, np.newaxis]

    # sort by alphabetical order
    labels_sorted = np.sort(labels, axis=0)

    for word in labels_sorted:
        if word[0] not in dictionary:
            dictionary[word[0]] = new_word_id
            new_word_id += 1

    for word in labels:
        words.append(dictionary[word[0]])

    one_hots = to_categorical(words)

    return dictionary, one_hots

class HandFeatures():

    @staticmethod
    def compute_distance_EF(thumb_EF, index_EF, middle_EF, ring_EF, pinky_EF, palm):

        # EF relative distance
        distance1 = np.linalg.norm((thumb_EF - index_EF), axis=1)[:, np.newaxis]
        distance2 = np.linalg.norm((index_EF - middle_EF), axis=1)[:, np.newaxis]
        distance3 = np.linalg.norm((middle_EF - ring_EF), axis=1)[:, np.newaxis]
        distance4 = np.linalg.norm((ring_EF - pinky_EF), axis=1)[:, np.newaxis]

        # EF to palm relative distance
        distance5 = np.linalg.norm((thumb_EF - palm), axis=1)[:, np.newaxis]
        distance6 = np.linalg.norm((index_EF - palm), axis=1)[:, np.newaxis]
        distance7 = np.linalg.norm((middle_EF - palm), axis=1)[:, np.newaxis]
        distance8 = np.linalg.norm((ring_EF - palm), axis=1)[:, np.newaxis]
        distance9 = np.linalg.norm((pinky_EF - palm), axis=1)[:, np.newaxis]

        distances = np.concatenate((distance1, distance2, distance3,
                                    distance4, distance5, distance6,
                                    distance7, distance8, distance9), axis=1)

        return distances

    @staticmethod
    def compute_forearm_coordinate(hand_radius, wrist, elbow, display = False):

        # Y: Line connecting ulna hand to the elbow (pointing proximally)
        Y = (elbow - wrist)
        Y_norm = np.linalg.norm(Y, axis=1)[:, np.newaxis].repeat(3, axis=1)
        Y = Y / Y_norm

        # X: Line perpendicular to the plane formed by ulna hand, radial hand and elbow (pointing forward)
        vec1 = (hand_radius - wrist)
        vec1_norm = np.linalg.norm(vec1, axis=1)[:, np.newaxis].repeat(3, axis=1)
        vec1 = vec1 / vec1_norm

        X = np.cross(Y, vec1)
        X_norm = np.linalg.norm(X, axis=1)[:, np.newaxis].repeat(3, axis=1)
        X = X / X_norm

        # Z: The line perpendicular to X and Y (pointing to the right)
        Z = np.cross(X, Y)
        Z_norm = np.linalg.norm(Z, axis=1)[:, np.newaxis].repeat(3, axis=1)
        Z = Z / Z_norm

        forearm_coordinate = np.concatenate([X[:, :, np.newaxis], Y[:, :, np.newaxis], Z[:, :, np.newaxis]], axis=2)

        if display:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            X = elbow[0, 0]
            Y = elbow[0, 1]
            Z = elbow[0, 2]

            ax.quiver(X, Y, Z, forearm_coordinate[0, 0, 0], forearm_coordinate[0, 0, 1], forearm_coordinate[0, 0, 2], color='r')
            ax.quiver(X, Y, Z, forearm_coordinate[1, 1, 0], forearm_coordinate[1, 1, 1], forearm_coordinate[1, 1, 2], color='b')
            ax.quiver(X, Y, Z, forearm_coordinate[2, 2, 0], forearm_coordinate[2, 2, 1], forearm_coordinate[2, 2, 2], color='gr')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

        return forearm_coordinate

    @staticmethod
    def compute_hand_coordinate(index_proximal, pinky_proximal, palm, display = False):

        # Z: Line connecting the index and pinky base (pointing right)
        Z = (index_proximal - pinky_proximal)
        print(Z, 'Z')
        Z_norm = np.linalg.norm(Z, axis=1)[:, np.newaxis].repeat(3, axis=1)
        Z = Z / Z_norm
        print(Z, 'ZZ')

        # X: Line perpendicular to the plane formed by index, pinky base and palm (pointing forward)
        vec1 = (index_proximal - palm)
        vec1_norm = np.linalg.norm(vec1, axis=1)[:, np.newaxis].repeat(3, axis=1)
        vec2 = vec1 / vec1_norm
        vec2 = (pinky_proximal - palm)
        vec2_norm = np.linalg.norm(vec2, axis=1)[:, np.newaxis].repeat(3, axis=1)
        vec2 = vec2 / vec2_norm

        X = np.cross(vec1, vec2)
        print(X, 'X')
        X_norm = np.linalg.norm(X, axis=1)[:, np.newaxis].repeat(3, axis=1)
        X = X / X_norm
        print(X, 'XX')

        # Y: The line perpendicular to Z and X (pointing proximally)
        Y = np.cross(Z, X)
        print(Y, 'Y')
        Y_norm = np.linalg.norm(Y, axis=1)[:, np.newaxis].repeat(3, axis=1)
        Y = Y / Y_norm
        print(Y, 'YY')

        hand_coordinate = np.concatenate([X[:, :, np.newaxis], Y[:, :, np.newaxis], Z[:, :, np.newaxis]], axis=2)

        if display:

            fig = plt.figure()
            ax = fig.gca(projection='3d')

            X = pinky_proximal[0, 0]
            Y = pinky_proximal[0, 1]
            Z = pinky_proximal[0, 2]

            ax.quiver(X, Y, Z, hand_coordinate[0, 0, 0], hand_coordinate[0, 0, 1], hand_coordinate[0, 0, 2], color='r')
            ax.quiver(X, Y, Z, hand_coordinate[1, 1, 0], hand_coordinate[1, 1, 1], hand_coordinate[1, 1, 2], color='b')
            ax.quiver(X, Y, Z, hand_coordinate[2, 2, 0], hand_coordinate[2, 2, 1], hand_coordinate[2, 2, 2], color='gr')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

        return hand_coordinate

    @staticmethod
    def compute_finger_joint_local_position(df_input, hand_c_L, hand_c_R):

        input = df_input.as_matrix()
        input_L = input[:, :input.shape[1]//2]
        input_R = input[:, input.shape[1]//2:]
        input_local_L = input_L.copy()
        input_local_R = input_R.copy()

        for t in range(input_L.shape[0]):

            temp_L = np.array([hand_c_L[t, :, 0], hand_c_L[t, :, 1], hand_c_L[t, :, 2]]).transpose()
            temp_R = np.array([hand_c_R[t, :, 0], hand_c_R[t, :, 1], hand_c_R[t, :, 2]]).transpose()

            for i in range(input_L.shape[1]//3):

                input_local_L[t, i*3:i*3+3] = np.dot(temp_L.transpose(), input_L[t, i*3:i*3+3])
                input_local_R[t, i*3:i*3+3] = np.dot(temp_R.transpose(), input_R[t, i*3:i*3+3])

        input_local = np.concatenate([input_local_L, input_local_R], axis=1)

        df_input_local = pd.DataFrame(input_local, columns=df_input.columns.values)

        thumb_local_LR = df_input_local.filter(like="thumb").as_matrix()
        index_local_LR = df_input_local.filter(like="index").as_matrix()
        middle_local_LR = df_input_local.filter(like="middle").as_matrix()
        ring_local_LR = df_input_local.filter(like="ring").as_matrix()
        pinky_local_LR = df_input_local.filter(like="pinky").as_matrix()

        locals = np.concatenate([thumb_local_LR,
                                 index_local_LR,
                                 middle_local_LR,
                                 ring_local_LR,
                                 pinky_local_LR])

        return locals

    @staticmethod
    def compute_finger_flexion_angle(EF, D, M, P, display = False):

        EF_D = EF - D
        D_M = D - M
        M_P = M - P

        EF_D_length = np.linalg.norm((EF_D), axis=1)[:, np.newaxis]
        D_M_length = np.linalg.norm((D_M), axis=1)[:, np.newaxis]
        M_P_length = np.linalg.norm((M_P), axis=1)[:, np.newaxis]

        angle1 = np.empty([EF.shape[0], 1])
        angle2 = np.empty([EF.shape[0], 1])

        for i in range(EF.shape[0]):
            angle1[i, 0] = np.arccos(
                np.dot(EF_D[i, :], D_M[i, :]) / (np.linalg.norm(EF_D[i, :]) * np.linalg.norm(D_M[i, :])))
            angle1[np.isnan(angle1)] = 0
            angle1[i, 0] = math.degrees(angle1[i, 0])

            angle2[i, 0] = np.arccos(
                np.dot(D_M[i, :], M_P[i, :]) / (np.linalg.norm(D_M[i, :]) * np.linalg.norm(M_P[i, :])))
            angle2[np.isnan(angle2)] = 0
            angle2[i, 0] = math.degrees(angle2[i, 0])

        if display:
            plt.plot(angle1, label='finger_angle_D_M')
            plt.plot(angle2, label='finger_angle_M_P')
            plt.legend()

        return angle1, angle2

    @staticmethod
    def compute_finger_abduction_angle(M, P, hand_c, display = False):

        M_P = P - M

        angle1 = np.empty([M_P.shape[0], 1])

        for i in range(M_P.shape[0]):
            # Express the vector (D_M) in the hand reference system
            R_hand_sub = np.array([hand_c[i, :, 0], hand_c[i, :, 1], hand_c[i, :, 2]])
            D_M_local = np.dot(R_hand_sub.transpose(), M_P[i, :])

            D_M_local_XZ = D_M_local.copy()
            D_M_local_XZ[1] = 0
            D_M_local_XZ = D_M_local_XZ / np.linalg.norm(D_M_local_XZ)

            R_hand_sub_Y = R_hand_sub[:, 1]

            angle1[i, 0] = np.arccos(
                np.dot(D_M_local_XZ, R_hand_sub_Y) / (np.linalg.norm(D_M_local_XZ) * np.linalg.norm(R_hand_sub_Y)))
            angle1[np.isnan(angle1)] = 0
            angle1[i, 0] = math.degrees(angle1[i, 0])

            '''
            ax.quiver(X, Y, Z, R_hand_sub[0, 0], R_hand_sub[0, 1], R_hand_sub[0, 2], color = 'r')
            ax.quiver(X, Y, Z, R_hand_sub[1, 0], R_hand_sub[1, 1], R_hand_sub[1, 2], color = 'b')
            ax.quiver(X, Y, Z, R_hand_sub[2, 0], R_hand_sub[2, 1], R_hand_sub[2, 2], color = 'gr')

            ax.quiver(X, Y, Z, D_M_local_XZ[0], D_M_local_XZ[1], D_M_local_XZ[2], color = 'k')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            '''

        if display:
            plt.plot(angle1, label='finger_angle_abduction')
            plt.legend()

        return angle1

    @staticmethod
    def compute_thumb_flexion_angle(EF, D, P, display = False):

        EF_D = EF - D
        D_P = D - P

        EF_D_length = np.linalg.norm((EF_D), axis=1)[:, np.newaxis]
        D_P_length = np.linalg.norm((D_P), axis=1)[:, np.newaxis]

        angle = np.empty([EF.shape[0], 1])

        for i in range(EF.shape[0]):
            angle[i, 0] = np.arccos(
                np.dot(EF_D[i, :], D_P[i, :]) / (np.linalg.norm(EF_D[i, :]) * np.linalg.norm(D_P[i, :])))
            angle[np.isnan(angle)] = 0
            angle[i, 0] = math.degrees(angle[i, 0])

        if display:
            plt.plot(angle, label='thumb_flexion_angle_D_P')

        return angle

    @staticmethod
    def compute_rotation_matrix(c1, c2, display=False, inverted=False):

        angle1, angle2, angle3 = np.empty([c1.shape[0], 1]), np.empty([c1.shape[0], 1]), np.empty([c1.shape[0], 1])

        for i in range(0, c1.shape[0]):
            c1_sub = np.array([c1[i, :, 0], c1[i, :, 1], c1[i, :, 2]]).transpose()
            c2_sub = np.array([c2[i, :, 0], c2[i, :, 1], c2[i, :, 2]])

            if inverted:
                c1_sub = np.array([c2[i, :, 0], c2[i, :, 1], c2[i, :, 2]]).transpose()
                c2_sub = np.array([c1[i, :, 0], c1[i, :, 1], c1[i, :, 2]])

            R = np.matmul(c1_sub, c2_sub)
            if R[1, 2] > 1:
                R[1, 2] = 1

            angle1[i, :] = math.asin(R[2, 1])
            angle2[i, :] = math.degrees(math.atan2(-R[2, 0] / math.cos(angle1[i, :]), R[2, 2] / math.cos(angle1[i, :])))
            angle3[i, :] = math.degrees(math.atan2(-R[0, 1] / math.cos(angle1[i, :]), R[1, 1] / math.cos(angle1[i, :])))

            angle1[i, :] = math.degrees(math.asin(R[2, 1]))

        if display:
            plt.plot(angle1, label='X')
            plt.plot(angle2, label='Y')
            plt.plot(angle3, label='Z')
            plt.legend()

        return angle1, angle2, angle3

    @staticmethod
    def plot_hands(df_input: pd.DataFrame):

        joint_names = [x[:-2] for x in df_input.columns.values]
        input = df_input.as_matrix()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i in range(2, input.shape[1]//3):

            x = input[0, i*3]
            y = input[0, i*3 + 1]
            z = input[0, i*3 + 2]

            ax.scatter(x, y, z)
            ax.text(x, y, z, joint_names[i*3])

    @staticmethod
    def display_hand_info(df_input: pd.DataFrame):

        pass
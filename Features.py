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

class Kinematics():

    coodinate_forearmR: np.ndarray
    coodinate_forearmL: np.ndarray
    coodinate_handR: np.ndarray
    coodinate_handL: np.ndarray

    def __init__(self, df_input: pd.DataFrame):

        self.df_input = df_input

        self.thumb_EF_R = df_input.values[:, :]
        self.thumb_D_R = df_input.values[:, :]
        self.thumb_P_R = df_input.values[:, :]

        self.index_EF_R = df_input.values[:, :]
        self.index_D_R = df_input.values[:, :]
        self.index_M_R = df_input.values[:, :]
        self.index_P_R = df_input.values[:, :]

        self.middle_EF_R = df_input.values[:, :]
        self.middle_D_R = df_input.values[:, :]
        self.middle_M_R = df_input.values[:, :]
        self.middle_P_R = df_input.values[:, :]

        self.ring_EF_R = df_input.values[:, :]
        self.ring_D_R = df_input.values[:, :]
        self.ring_M_R = df_input.values[:, :]
        self.ring_P_R = df_input.values[:, :]

        self.pinky_EF_R = df_input.values[:, :]
        self.pinky_D_R = df_input.values[:, :]
        self.pinky_M_R = df_input.values[:, :]
        self.pinky_P_R = df_input.values[:, :]

        self.thumb_EF_L = df_input.values[:, :]
        self.thumb_D_L = df_input.values[:, :]
        self.thumb_P_L = df_input.values[:, :]

        self.index_EF_L = df_input.values[:, :]
        self.index_D_L = df_input.values[:, :]
        self.index_M_L = df_input.values[:, :]
        self.index_P_L = df_input.values[:, :]

        self.middle_EF_L = df_input.values[:, :]
        self.middle_D_L = df_input.values[:, :]
        self.middle_M_L = df_input.values[:, :]
        self.middle_P_L = df_input.values[:, :]

        self.ring_EF_L = df_input.values[:, :]
        self.ring_D_L = df_input.values[:, :]
        self.ring_M_L = df_input.values[:, :]
        self.ring_P_L = df_input.values[:, :]

        self.pinky_EF_L = df_input.values[:, :]
        self.pinky_D_L = df_input.values[:, :]
        self.pinky_M_L = df_input.values[:, :]
        self.pinky_P_L = df_input.values[:, :]


        self.palm_R = df_input.values[:, :]
        self.palm_L = df_input.values[:, :]

        self.hand_radius_R = df_input.values[:, :]
        self.wrist_R = df_input.values[:, :]
        self.elbow_R = df_input.values[:, :]

        self.hand_radius_L = df_input.values[:, :]
        self.wrist_L = df_input.values[:, :]
        self.elbow_L = df_input.values[:, :]

        self._generate_coodinate()

    def _generate_distance(self):

        # distance between the end-effector and the end-effectors with the palm
        distanceR = HandFeatures.compute_distance_EF(thumb_EF=self.thumb_EF_R, index_EF=self.index_EF_R,
                                                     middle_EF=self.middle_EF_R, ring_EF=self.ring_EF_R,
                                                     pinky_EF=self.pinky_EF_R, palm=self.palm_R)

        distanceL = HandFeatures.compute_distance_EF(thumb_EF=self.thumb_EF_L, index_EF=self.index_EF_L,
                                                     middle_EF=self.middle_EF_L, ring_EF=self.ring_EF_L,
                                                     pinky_EF=self.pinky_EF_L, palm=self.palm_L)

        return distanceR, distanceL

    def _generate_coodinate(self):

        # Right Forearm reference system
        self.coodinate_forearmR = HandFeatures.compute_forearm_coordinate(hand_radius=self.hand_radius_R,
                                                                          wrist=self.wrist_R,
                                                                          elbow=self.elbow_R, display=False)

        # Left Forearm reference system
        self.coodinate_forearmL = HandFeatures.compute_forearm_coordinate(hand_radius=self.hand_radius_L,
                                                                          wrist=self.wrist_L,
                                                                          elbow=self.elbow_L, display=False)

        # Right Hand reference system
        self.coodinate_handR = HandFeatures.compute_hand_coordinate(index_proximal=self.index_M_R,
                                                                    pinky_proximal=self.pinky_M_R,
                                                                    palm=self.palm_R, display=False)

        # Left Hand reference system
        self.coodinate_handL = HandFeatures.compute_hand_coordinate(index_proximal=self.index_M_L,
                                                                    pinky_proximal=self.pinky_M_L,
                                                                    palm=self.palm_L, display=False)

    def _generate_rotation_matrix(self):

        coodinate_forearmR, coodinate_forearmL, coodinate_handR, coodinate_handL = self._compute_coodinate()

        # Rotation matrix and angle between the two forearms
        angle1, angle2, angle3 = HandFeatures.compute_rotation_matrix(c1=coodinate_forearmR, c2=coodinate_forearmL,
                                                                      display=False)
        angle1[:] = 0
        angle2[:] = 0
        angle3[:] = 0

        # Rotation matrix and angle between the elbow and the forearm right
        angle4, angle5, angle6 = HandFeatures.compute_rotation_matrix(c1=coodinate_forearmR, c2=coodinate_handR,
                                                                      display=False)

        # Rotation matrix and angle between the elbow and the forearm left
        angle7, angle8, angle9 = HandFeatures.compute_rotation_matrix(c1=coodinate_forearmL, c2=coodinate_handL,
                                                                      display=False)

        angle_rotation = np.concatenate([angle1, angle2, angle3,
                                         angle4, angle5, angle6,
                                         angle7, angle8, angle9])

        return angle_rotation

    def _generate_flexion_angle(self):

        thumb_flexionR = HandFeatures.compute_thumb_flexion_angle(EF=self.thumb_EF_R,
                                                                  D=self.thumb_D_R,
                                                                  P=self.thumb_P_R, display=False)

        index_flexionR1, index_flexionR2 = HandFeatures.compute_finger_flexion_angle(EF=self.index_EF_R,
                                                                                     D=self.index_D_R,
                                                                                     M=self.index_M_R,
                                                                                     P=self.index_P_R, display=False)

        middle_flexionR1, middle_flexionR2 = HandFeatures.compute_finger_flexion_angle(EF=self.middle_EF_R,
                                                                                       D=self.middle_D_R,
                                                                                       M=self.middle_M_R,
                                                                                       P=self.middle_P_R, display=False)

        ring_flexionR1, ring_flexionR2 = HandFeatures.compute_finger_flexion_angle(EF=self.ring_EF_R,
                                                                                   D=self.ring_D_R,
                                                                                   M=self.ring_M_R,
                                                                                   P=self.ring_P_R, display=False)

        pinky_flexionR1, pinky_flexionR2 = HandFeatures.compute_finger_flexion_angle(EF=self.pinky_EF_R,
                                                                                     D=self.pinky_D_R,
                                                                                     M=self.pinky_M_R,
                                                                                     P=self.pinky_P_R, display=False)

        fingers_flexionR = np.concatenate((thumb_flexionR,
                                           index_flexionR1, index_flexionR2,
                                           middle_flexionR1, middle_flexionR2,
                                           ring_flexionR1, ring_flexionR2,
                                           pinky_flexionR1, pinky_flexionR2), axis=1)


        thumb_flexionL = HandFeatures.compute_thumb_flexion_angle(EF=self.thumb_EF_L,
                                                                  D=self.thumb_D_L,
                                                                  P=self.thumb_P_L, display=False)

        index_flexionL1, index_flexionL2 = HandFeatures.compute_finger_flexion_angle(EF=self.index_EF_L,
                                                                                     D=self.index_D_L,
                                                                                     M=self.index_M_L,
                                                                                     P=self.index_P_L, display=False)

        middle_flexionL1, middle_flexionL2 = HandFeatures.compute_finger_flexion_angle(EF=self.middle_EF_L,
                                                                                       D=self.middle_D_L,
                                                                                       M=self.middle_M_L,
                                                                                       P=self.middle_P_L, display=False)

        ring_flexionL1, ring_flexionL2 = HandFeatures.compute_finger_flexion_angle(EF=self.ring_EF_L,
                                                                                   D=self.ring_D_L,
                                                                                   M=self.ring_M_L,
                                                                                   P=self.ring_P_L, display=False)

        pinky_flexionL1, pinky_flexionL2 = HandFeatures.compute_finger_flexion_angle(EF=self.pinky_EF_L,
                                                                                     D=self.pinky_D_L,
                                                                                     M=self.pinky_M_L,
                                                                                     P=self.pinky_P_L, display=False)

        fingers_flexionL = np.concatenate((thumb_flexionL,
                                           index_flexionL1, index_flexionL2,
                                           middle_flexionL1, middle_flexionL2,
                                           ring_flexionL1, ring_flexionL2,
                                           pinky_flexionL1, pinky_flexionL2), axis=1)

        fingers_flexion = np.concatenate((fingers_flexionR, fingers_flexionL), axis=1)

        return fingers_flexion

    def _generate_abduction_angle(self):

        index_abductionR = HandFeatures.compute_finger_abduction_angle(M=self.index_M_R, P=self.index_P_R,
                                                                       hand_c=self.coodinate_handR, display=False)
        middle_abductionR = HandFeatures.compute_finger_abduction_angle(M=self.middle_M_R, P=self.middle_P_R,
                                                                        hand_c=self.coodinate_handR, display=False)
        ring_abductionR = HandFeatures.compute_finger_abduction_angle(M=self.ring_M_R, P=self.ring_P_R,
                                                                      hand_c=self.coodinate_handR, display=False)
        pinky_abductionR = HandFeatures.compute_finger_abduction_angle(M=self.pinky_M_R, P=self.pinky_P_R,
                                                                       hand_c=self.coodinate_handR, display=False)

        fingers_abductionR = np.concatenate((index_abductionR, middle_abductionR, ring_abductionR, pinky_abductionR),
                                            axis=1)

        index_abductionL = HandFeatures.compute_finger_abduction_angle(M=self.index_M_L, P=self.index_P_L,
                                                                       hand_c=self.coodinate_handL, display=False)
        middle_abductionL = HandFeatures.compute_finger_abduction_angle(M=self.middle_M_L, P=self.middle_P_L,
                                                                        hand_c=self.coodinate_handL, display=False)
        ring_abductionL = HandFeatures.compute_finger_abduction_angle(M=self.ring_M_L, P=self.ring_P_L,
                                                                      hand_c=self.coodinate_handL, display=False)
        pinky_abductionL = HandFeatures.compute_finger_abduction_angle(M=self.pinky_M_L, P=self.pinky_P_L,
                                                                       hand_c=self.coodinate_handL, display=False)

        fingers_abductionL = np.concatenate((index_abductionL, middle_abductionL, ring_abductionL, pinky_abductionL),
                                            axis=1)

        fingers_abduction = np.concatenate((fingers_abductionR, fingers_abductionL), axis=1)

        return fingers_abduction

    def generate_kinematics(self, display):

        distanceR, distanceL = self._generate_distance()

        angle_rotation = self._generate_rotation_matrix()

        # Computing of thumb and fingers angle (2 angles for the thumb and 3 angles for each finger)
        angle_fingers_flexion = self._generate_flexion_angle()

        angle_fingers_abduction = self._generate_abduction_angle()

        kinematics = np.concatenate((distanceR,
                                     distanceL,
                                     angle_rotation,
                                     angle_fingers_flexion,
                                     angle_fingers_abduction), axis=1)

        if display:
            print(kinematics[95:100, 0], "data kinematics")
            print(kinematics.shape, "data kinematics shape")

        return kinematics

    def generate_locals(self, display):

        locals = HandFeatures.compute_finger_joint_local_position(df_input=self.df_input,
                                                                  hand_c_L=self.coodinate_handL,
                                                                  hand_c_R=self.coodinate_handR)

        if display:
            print(locals, "locals")
            print(locals.shap)

        return locals
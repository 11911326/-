import numpy as np
import matplotlib.pyplot as plt

al_max = 10
ar_max = 10
channel_state = [1, 2]
channel_prob = [0.5, 0.5]
channel_number = len(channel_state)


def action_result(al, ar, action):
    list_after = []
    if action == 1:
        age_device = min(al + 1, al_max)
        age_destination = min(ar + 1, ar_max)
    elif action == 2:
        age_device = min(al + 1, al_max)
        age_destination = min(al + 1, ar_max)
    elif action == 3:
        age_device = 1
        age_destination = min(ar + 1, ar_max)
    else:
        age_device = 1
        age_destination = min(al + 1, ar_max)
    list_after.append(age_device)
    list_after.append(age_destination)
    return list_after


def cost_lagrange(age_destination, action, channel):
    if action == 1:
        cost = 0
    elif action == 2:
        cost = 3.5 / channel
    elif action == 3:
        cost = 2
    else:
        cost = 2 + 3.5 / channel
    return age_destination + lam * cost


def policy_evaluation(Value_matrix, itration_time):
    itration_time_plus = itration_time + 1
    V_out = np.zeros((ar_max, al_max, channel_number), dtype=float)
    if itration_time < 100:
        for ar in range(0, ar_max):
            for al in range(0, al_max):
                J_cost = []
                J_refer = []
                for ac in range(0, 4):
                    for ch in range(0, channel_number):
                        change = action_result(al + 1, ar + 1, ac + 1)
                        change_refer = action_result(1, 1, ac + 1)
                        J_cost.append(
                            cost_lagrange(ar + 1, ac + 1, channel_state[ch]) + value_estimate(Value_matrix,
                                                                                              change[0] - 1,
                                                                                              change[1] - 1))
                        J_refer.append(cost_lagrange(1, ac + 1, channel_state[ch]) + value_estimate(Value_matrix,
                                                                                                    change_refer[0] - 1,
                                                                                                    change_refer[1] - 1))
                V_out[ar, al, ch] = min(J_cost) - min(J_refer)

        policy_evaluation(V_out, itration_time_plus)
        return V_out
    if itration_time == 100:
        return Value_matrix


def policy_improvement(Value_matrix):
    V_policy = np.zeros((ar_max, al_max, channel_number), dtype=float)
    for ar in range(0, ar_max):
        for al in range(0, al_max):
            J_cost = []
            for ac in range(0, 4):
                for ch in range(0, channel_number):
                    change = action_result(al + 1, ar + 1, ac + 1)
                    J_cost.append(
                        cost_lagrange(ar + 1, ac + 1, channel_state[ch]) + value_estimate(Value_matrix, change[0] - 1,
                                                                                         change[1] - 1))
                    V_policy[ar][al][ch] = J_cost.index(min(J_cost)) + 1

    return V_policy


def value_estimate(Value_matrix, A_l, A_r):
    Estimation = 0
    for ch in range(0, channel_number):
        Estimation = Estimation + Value_matrix[A_r, A_l, ch] * channel_prob[ch]
    return Estimation


if __name__ == '__main__':
    lam = 1
    V_0 = np.zeros((ar_max, al_max, channel_number), dtype=float)
    V_result = policy_evaluation(V_0, 0)
    print(V_result)
    policy_result = policy_improvement(V_result)
    print(policy_result)
    for h in range(0, channel_number):
        plt.figure(h + 1)
        plt.xlim(xmax=11, xmin=0)
        plt.ylim(ymax=11, ymin=0)
        plt.xlabel("AOI at the device")
        plt.ylabel("AOI at the destination")
        for i in range(0, 10):
            for j in range(0, 10):
                if policy_result[i][j][h] == 1:
                    plt.plot([j + 1], [i + 1], marker='D', color='g')
                elif policy_result[i][j][h] == 2:
                    plt.plot([j + 1], [i + 1], marker='^', color='b')
                elif policy_result[i][j][h] == 3:
                    plt.plot([j + 1], [i + 1], marker='o', color='r')
                elif policy_result[i][j][h] == 4:
                    plt.plot([j + 1], [i + 1], marker='x', color='y')

    plt.show()

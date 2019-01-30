#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LCA-QLBS.py
Author: M. Ohmori
"""

import numpy as np
import random
import math
#from iexfinance.stocks import get_historical_data
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
from statistics import mean, median, variance, stdev
import csv
import copy
import os

L = 20  #league size 20 8
L_half = L / 2  #計算速度向上の為
episode = 1000  #season 1000 38
#n = 10         #次元数
PSI1 = 0.2
PSI2 = 1.0
schedule = list()
period = 25  # 25
MaxHanpuku = 10000

sigma = 0.15  #volatility
mu = 0.05  #drift
gamma = 0.2  #割引率
LAMBDA = 0.001  #risk aversion parameter
r = 0.03  #無リスク利子率
delta_t = 1.0 / 24.0


class LeagueChampionshipAlgorithm(object):
    def __init__(self):
        return

    def league(self):
        # ファイルオープン
        date = datetime.now()
        f2 = open('LCA-QLBS-DATA{0:%Y%m%d_%H%M%S}.csv'.format(date), 'w')
        writer = csv.writer(f2, lineterminator='\n')
        # データをリストに保持
        csvlist = []
        #csvlist.append("S, X, a, PI, R, Q")
        path_w = 'LCA-QLBSshutsuryoku.txt'
        wfile = open(path_w, mode='w')

        global L, L_half

        X = self.getRandomTeam()  #モデルの変数、現状態
        S = np.array([[100 for i in range(period)] for l in range(L)])  #stock
        deltaS = np.copy(S)

        flag = 0
        if L % 2 == 1:  #チーム数が奇数の場合、ダミーチームを作る
            DAMY_TEAM = [1000 for step in range(period)]
            X.append(DAMY_TEAM.copy())
            L += 1
            L_half += 1
            flag = 1
        """
        start = datetime(2014, 8, 20)
        end = start + timedelta(days=period)
        name = self.symbolName()
        """

        q_table = self.getQ_table()  #Q value table

        fX = self.optimizationFunction(q_table)  #適応度
        #fX = self.optimizationFunction(fX)
        nextX = np.copy(X)  #次状態
        B = np.copy(X[0])  #再良解
        fB = np.copy(fX[0])  #最適な適応度
        f_best = np.max(fB)  #適応度の最も良い値
        f_bList = np.empty(0)  #f_bestのリスト

        n = 0
        schedule = self.leagueSchedule(n)  #schedule[週][チーム番号]
        wfile.write("n =%d, f_best=%f\n" % (n, f_best))
        #for stockSymbol in range(len(name)):
        R = np.round(np.random.random(L), decimals=6)

        action = np.arange(0, 1, 0.01)

        PI = np.array(np.round(((100 - 0) * np.random.rand(period) + 0), decimals=6))  #ポートフォリオΠ
        """
        for stockSymbol in range(1):
            stock = np.array(
                get_historical_data(
                    name[stockSymbol], start,
                    output_format='pandas')['close'][:period])
            X = np.copy(stock)
        """

        for hanpuku in range(MaxHanpuku):
            t = 0
            while t < period:
                n = 0
                while n < episode * (L - 1):
                    if flag == 1 and n != 0:
                        X.append(DAMY_TEAM.copy())
                        state = X[t]

                    for l in range(L):
                        #print(t, n, l)
                        S[l][t] = np.round(
                            (np.exp(X[l][t] + (mu - (sigma**2 / 2) * t))),
                            decimals=6)
                        #print(S[l][t])
                        Act = self.get_action(q_table[l][t])
                        if t == period - 1:
                            Act = 0

                        Time = period - 1
                        while Time > t:
                            Time += -1
                            deltaS[l][Time] = S[l][Time + 1] - np.exp(
                                r * delta_t) * S[l][Time]
                            deltaS[l][Time] = np.round(deltaS[l][Time],
                                                       6)  #桁丸め
                            PI[Time] = np.exp(-r * delta_t) * (
                                PI[Time + 1] - action[Act] * deltaS[l][Time])

                        hatPI = self.getHat(PI, t + 1)
                        hatS = self.getHat(S[l], t)
                        if t == period - 1:
                            R = LAMBDA * np.var(PI)
                            R = round(R, 6)
                            q_table[l][t][Act] = -PI[t] - LAMBDA * np.var(PI)
                        else:
                            R = gamma * action[Act] * deltaS[l][t] - LAMBDA * gamma**2 * np.mean(
                                hatPI**2 - 2 * action[Act] * hatS * hatPI +
                                action[Act]**2 * hatS**2)
                            R = round(R, 6)
                            q_table[l][t][Act] = np.mean(
                                R + gamma * max(q_table[l][t + 1]))

                    #print(t, n, l, PI)

                    #self.backQ(X, currentNode)

                    Y = self.get_Y()
                    for l in range(L - 1):
                        teamA, teamB, teamC, teamD = self.teamClassification(
                            X, t, l - 1, n)
                        winner1 = self.winORlose(t, X, teamA, teamB, fX,
                                                 f_best)
                        winner2 = self.winORlose(t, X, teamC, teamD, fX,
                                                 f_best)
                        C = self.setTeamFormation(t, X, B, Y, teamA, teamB,
                                                  teamC, teamD, winner1,
                                                  winner2)
                        nextX[np.where(X == teamA)[0][0]] = np.copy(C)

                    X = np.copy(nextX)
                    if flag == 1:
                        del X[-1]  #DAMY_TEAMの削除
                    fX = self.optimizationFunction(q_table)  #適応度
                    for l in range(L):
                        if fX[l][t] > fB[t]:
                            B[t] = X[l][t]
                            fB[t] = fX[l][t]
                    f_best = max(fB)
                    f_bList = np.append(f_bList, f_best)

                    if n % 10 == 0:
                        wfile.write("t =%d, f_best=%f\n" % (n, f_best))

                    if n % (L - 1) == 0:
                        #self.addOnModule() #add-onを追加できる
                        schedule = self.leagueSchedule(n)

                    n += 1
                print(hanpuku, t, n, l)
                t += 1

            # Data Save
            csvlist.append(S[0])
            csvlist.append(X[0])
            #csvlist.append(action)
            csvlist.append(PI)
            #csvlist.append(R)
            #csvlist.append(q_table)

            # 出力
            writer.writerow(csvlist)

        wfile.write("~~~~出力結果~~~~\n")
        #f.write("{x: new_x %f y: new_y %f}\n" % (fB[0], fB[1]))
        wfile.write("f(x,y) = %f\n" % f_best)
        self.shutsuryoku(fB, f_best)

        # ファイルクローズ
        f2.close()
        wfile.close()
        return S[0], X[0], PI, fB

    def getRandomTeam(self):
        #チーム個体の初期値
        X = np.array(
            [[round(random.uniform(10.0, 20.0), 6) for i in range(period)]
             for l in range(L)])  #6桁に丸める
        return X

    def symbolName(self):
        name = [
            "AAPL", "TM", "MARK", "INFO", "PCG", "AMD", "GG", "GE", "BAC",
            "MU", "CHK", "C", "F", "ABEV", "BABA", "PEP", "YUM", "FB", "NFLX",
            "USO"
        ]
        return name

    def get_action(self, q_table):
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001
        if epsilon < np.random.uniform(0, 1):  # 0以上1未満の一様乱数を1個生成
            next_Act = np.argmax(q_table)
        else:
            next_Act = np.random.randint(0, 100)
        return next_Act

    def getHat(self, S, t):
        barS = S[:t + 1].mean()
        hatS = S - barS
        return hatS

    def getQ_table(self):
        q_table = np.round(
            np.random.uniform(0, 1, (L, period, 100)), decimals=6)
        #100:株式保有量（率）
        #25*30×100(action)×Lのq_table
        #for team in range(L)]  #6桁に丸める
        return q_table

    def leagueSchedule(self, t):
        #リーグスケジュールの設定
        if t == 1:
            schedule.append([l + 1 for l in range(L - 1)])

        randSche = random.sample([l + 1 for l in range(L - 1)], L - 1)
        schedule.append(randSche.copy())

        for l in range(L - 2):
            randSche.append(randSche.pop(0))
            schedule.append(randSche.copy())
        return schedule

    def teamClassification(self, X, t, l, n):
        if l == -1:
            teamA = X[0][t]
            teamB = X[schedule[t][l + 1]][t]
            teamC = X[schedule[t][l + 2]][t]
            teamD = X[schedule[t][L - 1 - (l + 2)]][t]
        elif l == 0:
            teamA = X[schedule[t][l]][t]
            teamB = X[0][t]
            teamC = X[schedule[t][l + 2]][t]
            teamD = X[schedule[t][L - 1 - (l + 2)]][t]
        elif l == 1:
            teamA = X[schedule[t][l]][t]
            teamB = X[schedule[t][L - 1 - l]][t]
            teamC = X[0][t]
            teamD = X[schedule[t][l - 1]][t]
        elif l == 2:
            teamA = X[schedule[t][l]][t]
            teamB = X[schedule[t][L - 1 - l]][t]
            teamC = X[schedule[t][l - 2]][t]
            teamD = X[0][t]
        elif l < L_half:
            teamA = X[schedule[t][l]][t]
            teamB = X[schedule[t][L - 1 - l]][t]
            teamC = X[schedule[t][L - l + 1]][t]
            teamD = X[schedule[t][l - 2]][t]
        elif l == L_half - 1:
            teamA = X[schedule[t][l]][t]
            teamB = X[schedule[t][l + 1]][t]
            teamC = X[schedule[t][l + 3]][t]
            teamD = X[schedule[t][l - 2]][t]
        elif l == L_half:
            teamA = X[schedule[t][l]][t]
            teamB = X[schedule[t][l - 1]][t]
            teamC = X[schedule[t][l + 1]][t]
            teamD = X[schedule[t][l - 2]][t]
        elif l == L_half + 1:
            teamA = X[schedule[t][l]][t]
            teamB = X[schedule[t][l - 3]][t]
            teamC = X[schedule[t][l - 1]][t]
            teamD = X[schedule[t][l - 2]][t]
        elif l == L_half + 2:
            teamA = X[schedule[t][l]][t]
            teamB = X[schedule[t][l - 5]][t]
            teamC = X[schedule[t][l - 3]][t]
            teamD = X[schedule[t][l - 2]][t]
        elif l < L - 1:
            teamA = X[schedule[t][l]][t]
            teamB = X[schedule[t][L - l - 1]][t]
            teamC = X[schedule[t][L - l + 1]][t]
            teamD = X[schedule[t][l - 2]][t]
        return teamA, teamB, teamC, teamD

    def optimizationFunction(self, Q):
        "fitness = sum_sell - sum_buy"
        #fX = sum_sell - sum_buy
        return np.amax(Q, axis=2)  #Q最大のaction

    def winORlose(self, t, X, team1, team2, fX, f_best):
        Index1 = np.where(X == team1)[0][0]
        Index2 = np.where(X == team2)[0][0]
        winPoint = (fX[Index2][t] - f_best) / (
            fX[Index2][t] + fX[Index1][t] - 2.0 * f_best)

        shouritsu = random.uniform(0.0, 1.0)
        shouritsu = np.round(shouritsu, decimals=6)
        #勝敗
        if winPoint <= 0.5:
            if winPoint == 0.0:
                winner = team2
            elif shouritsu <= winPoint:
                winner = team1
            else:
                winner = team2
        else:
            if winPoint == 1.0:
                winner = team1
            elif winPoint <= shouritsu:
                winner = team2
            else:
                winner = team1
        return winner

    def get_Y(self):
        q0 = 1  #q_0=1   #フラグ本数
        Y = list()  #バイナリ変数配列
        for i in range(L):
            a = random.uniform(0.0, 1.0)

            poInt = [0]
            aA = random.uniform(0.0, 1.0)
            if aA < a:
                poInt[0] = 1
            #poInt = list(random.sample([1 for l in range(len(y_sample))],q))  #どれを1にするか
            #poInt.sort()
            Y.append(poInt)
        return Y

    def getRandom_rid(self):
        r_id1 = [round(random.uniform(0.0, 1.0), 6) for i in range(period)]
        r_id2 = [round(random.uniform(0.0, 1.0), 6) for i in range(period)]
        return r_id1, r_id2

    def addOnModule(self):
        return

    def setTeamFormation(self, t, X, B, Y, teamA, teamB, teamC, teamD, winner1,
                         winner2):
        r_id1, r_id2 = self.getRandom_rid()
        X2 = np.copy(X)
        nextX = np.copy(X2)
        lA = np.where(X == teamA)[0][0]
        lB = np.where(X == teamB)[0][0]
        lC = np.where(X == teamC)[0][0]
        lD = np.where(X == teamD)[0][0]
        if winner1 == teamA and winner2 == teamC:  #S/T
            nextX[lA][t] = B[t] + Y[lA][0] * (
                PSI1 * r_id1[t] * (X2[lA][t] - X2[lD][t]) + PSI1 * r_id2[t] *
                (X2[lA][t] - X2[lB][t]))
        elif winner1 == teamA and winner2 == teamD:  #S/O
            nextX[lA][t] = B[t] + Y[lA][0] * (
                PSI2 * r_id1[t] * (X2[lD][t] - X2[lA][t]) + PSI1 * r_id2[t] *
                (X2[lA][t] - X2[lB][t]))
        elif winner1 == teamB and winner2 == teamC:  #W/T
            nextX[lA][t] = B[t] + Y[lA][0] * (
                PSI1 * r_id1[t] * (X2[lA][t] - X2[lD][t]) + PSI2 * r_id2[t] *
                (X2[lB][t] - X2[lA][t]))
        elif winner1 == teamB and winner2 == teamD:  #W/O
            nextX[lA][t] = B[t] + Y[lA][0] * (
                PSI2 * r_id1[t] * (X2[lD][t] - X2[lA][t]) + PSI2 * r_id2[t] *
                (X2[lB][t] - X2[lA][t]))
        return np.round(nextX[lA], decimals=6)

    def shutsuryoku(self, fB, f_best):
        print("~~~~出力結果~~~~")
        #print("{x: new_x", fB[0], "y: new_y", fB[1], "}")
        print("f(x,y) = ", f_best)
        return


if __name__ == "__main__":

    new_dir_path_recursive = 'LCA-QLBSPlot'

    os.makedirs(new_dir_path_recursive, exist_ok=True)

    LCA = LeagueChampionshipAlgorithm()
    date = datetime.now()
    yS, yX, yPI, yfB = LCA.league()

    x = list(t for t in range(period))

    pf = plt.figure()
    #plt.subplot(2, 2, 1)
    plt.plot(x, yS, label="S")
    plt.xlabel("Time Step")
    plt.title("Simulated stock prtice S")
    pf.savefig('LCA-QLBSPlot/LCA-QLBS{0:%Y%m%d_%H%M%S}_S.pdf'.format(date))
    #plt.subplot(2, 2, 2)
    pf = plt.figure()
    plt.plot(x, yX, label="X")
    plt.xlabel("Time Step")
    plt.title("State variable X")
    pf.savefig('LCA-QLBSPlot/LCA-QLBS{0:%Y%m%d_%H%M%S}_X.pdf'.format(date))
    #plt.subplot(2, 2, 3)
    pf = plt.figure()
    plt.plot(x, yPI, label="Π")
    plt.xlabel("Time Step")
    plt.title("Optimal portfolio Π")
    pf.savefig('LCA-QLBSPlot/LCA-QLBS{0:%Y%m%d_%H%M%S}_PI.pdf'.format(date))
    #plt.subplot(2, 2, 4)
    pf = plt.figure()
    plt.plot(x, yfB, label="Q")
    plt.xlabel("Time Step")
    plt.title("Optimal LCA Q-function Q")
    pf.savefig('LCA-QLBSPlot/LCA-QLBS{0:%Y%m%d_%H%M%S}_Q.pdf'.format(date))

    #plt.ylabel("Position")

    plt.show()
    #pf.savefig('LCA-QLBSPlot/LCA-QLBS{0:%Y%m%d_%H%M%S}.pdf'.format(date))

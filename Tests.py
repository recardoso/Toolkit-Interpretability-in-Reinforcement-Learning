import numpy as np
import random
from random import randint
import collections
from collections import Counter 
import matplotlib.pyplot as plt
from tabulate import tabulate

#self made files
from explanations import *

class example1():
    def __init__(self):
        Pl = np.zeros((4,1,4))
        Pl[0,0,1] = 0.5
        Pl[0,0,2] = 0.5
        Pl[1,0,3] = 1
        Pl[2,0,2] = 0.5
        Pl[2,0,3] = 0.5


        Rl = np.zeros((4,1))
        Rl[0,0] = 0
        Rl[1,0] = -500
        Rl[2,0] = 0
        Rl[3,0] = 1000

        goal = [3]
    
        self.expl = Explanations(4,Pl,Rl,1,0.90,goal)

    
    def test1(self):
        self.expl.single_get_state_explanation_distribution(0)

    def test2(self):
        self.expl.single_get_state_explanation_distribution(0,change_Pl=False)

    def test3(self):
        self.expl.c51_state_distribution(0,-1000,1000)

    def test4(self):
        self.expl.c51_state_distribution(0,-1000,1000, proc_actions=[0])

    def test5(self):
        path = [0,2,3]
        pathactions = [0,0]
        self.expl.get_path_distribution(path,pathactions)

    def test6(self):
        path = [0,2,3]
        pathactions = [0,0]
        bif = self.expl.get_path_bifurcations_outside_path(path,pathactions)

    def test7(self):
        bif = self.expl.get_state_bifurcations(0,0)

    def test8(self):
        pth = self.expl.get_optimal_path_from_state(0)
        print(pth)

    def test9(self):
        path = [0,2,3]
        pathactions = [0,0]
        self.expl.get_path_security_distribution(path,pathactions)
    
    def test10(self):
        path = [0,2,3]
        pathactions = [0,0]
        bif = self.expl.get_path_all_bifurcations(path,pathactions)


class example2():
    def __init__(self):
        # cliffwalk
        Pl = np.zeros((15,4,15))
        Pl[0,0,5] = 1
        Pl[0,3,1] = 1
        Pl[1,0,6] = 1
        Pl[1,2,0] = 1
        Pl[1,3,2] = 1
        Pl[2,0,7] = 1
        Pl[2,2,1] = 1
        Pl[2,3,3] = 1
        Pl[3,0,8] = 1
        Pl[3,2,2] = 1
        Pl[3,3,4] = 1
        Pl[4,0,9] = 1
        Pl[4,2,3] = 1
        Pl[5,0,10] = 1
        Pl[5,1,0] = 1
        Pl[5,3,6] = 1
        Pl[6,0,11] = 1
        Pl[6,1,1] = 1
        Pl[6,2,5] = 1
        Pl[6,3,7] = 1
        Pl[7,0,12] = 1
        Pl[7,1,2] = 1
        Pl[7,2,6] = 1
        Pl[7,3,8] = 1
        Pl[8,0,13] = 1
        Pl[8,1,3] = 1
        Pl[8,2,7] = 1
        Pl[8,3,9] = 1
        Pl[9,0,14] = 1
        Pl[9,1,4] = 1
        Pl[9,2,8] = 1
        Pl[10,1,5] = 1
        Pl[10,3,11] = 1
        Pl[11,1,6] = 1
        Pl[11,2,10] = 1
        Pl[11,3,12] = 1
        Pl[12,1,7] = 1
        Pl[12,2,11] = 1
        Pl[12,3,13] = 1
        Pl[13,1,8] = 1
        Pl[13,2,12] = 1
        Pl[13,3,14] = 1
        Pl[14,1,9] = 1
        Pl[14,2,13] = 1

        Rl = np.zeros((15,4))
        Rl[1,0] = -100
        Rl[1,1] = -100
        Rl[1,2] = -100
        Rl[1,3] = -100
        Rl[2,0] = -100
        Rl[2,1] = -100
        Rl[2,2] = -100
        Rl[2,3] = -100
        Rl[3,0] = -100
        Rl[3,1] = -100
        Rl[3,2] = -100
        Rl[3,3] = -100
        Rl[4,0] = 100
        Rl[4,1] = 100
        Rl[4,2] = 100
        Rl[4,3] = 100

        self.expl = Explanations(15,Pl,Rl,4,0.90,[4])


    def test1(self):
        self.expl.single_get_state_explanation_distribution(5,labels=['Action UP','Action DOWN','Action LEFT','Action RIGHT'])

    def test2(self):
        self.expl.single_get_state_explanation_distribution(5,change_Pl=False,labels=['Action UP','Action DOWN','Action LEFT','Action RIGHT'])

    def test3(self):
        self.expl.c51_state_distribution(0,-400,400)

    def test4(self):
        self.expl.c51_state_distribution(0,-400,400, proc_actions=[0])

    def test5(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        self.expl.get_path_distribution(path,pathactions)

    def test6(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        bif = self.expl.get_path_bifurcations_outside_path(path,pathactions)

    def test7(self):
        bif = self.expl.get_state_bifurcations(0,0)

    def test8(self):
        pth = self.expl.get_optimal_path_from_state(0)
        print(pth)

    def test9(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        self.expl.get_path_security_distribution(path,pathactions)

    def test10(self):
        self.expl.single_get_state_explanation_distribution(0,proc_actions=[0,3])

    def test11(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        self.expl.get_path_security_distribution(path,pathactions,epsilon=0.5,always_return_path=True,return_path_if_same_or_new=False,Never_return_path=False)
    
    def test12(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        self.expl.get_path_security_distribution(path,pathactions,epsilon=0.5,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False)

    def test13(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        self.expl.get_path_security_distribution(path,pathactions,epsilon=0.5,always_return_path=False,return_path_if_same_or_new=False,Never_return_path=True)

    def test14(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        self.expl.get_path_security_distribution(path,pathactions,epsilon=0.0,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False)

    def test15(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        self.expl.get_path_security_distribution(path,pathactions,epsilon=0.3,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False)

    def test16(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        self.expl.get_path_security_distribution(path,pathactions,epsilon=0.5,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False)

    def test17(self):
        path = [0,5,6,7,8,9,4]
        pathactions = [0,3,3,3,3,1]
        self.expl.get_path_security_distribution(path,pathactions,epsilon=0.7,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False)

class example3():
    def __init__(self):
        # cliffwalk
        Pl = np.zeros((4,2,4))
        Pl[0,0,1] = 1
        Pl[0,1,2] = 1
        Pl[1,1,3] = 1
        Pl[2,0,3] = 1

        Rl = np.zeros((4,2))
        Rl[3,0] = 100
        Rl[3,1] = 100

        self.expl = Explanations(4,Pl,Rl,2,0.90,[3])

    def test8(self):
        pth = self.expl.get_optimal_path_from_state(0)
        print(pth)

class example4():
    def __init__(self):
        # cliffwalk
        Pl = np.zeros((100,4,100))
        Pl[0,0,0] = 1.0
        Pl[0,1,10] = 1.0
        Pl[0,2,0] = 1.0
        Pl[0,3,0] = 0.5
        Pl[0,3,1] = 0.5
        Pl[1,0,1] = 1.0
        Pl[1,1,11] = 1.0
        Pl[1,2,0] = 0.5
        Pl[1,2,1] = 0.5
        Pl[1,3,2] = 1.0
        Pl[2,0,0] = 1.0
        Pl[2,1,0] = 1.0
        Pl[2,2,0] = 1.0
        Pl[2,3,0] = 1.0
        Pl[3,0,3] = 1.0
        Pl[3,1,13] = 1.0
        Pl[3,2,2] = 1.0
        Pl[3,3,4] = 1.0
        Pl[4,0,4] = 1.0
        Pl[4,1,4] = 0.5
        Pl[4,1,14] = 0.5
        Pl[4,2,3] = 1.0
        Pl[4,3,5] = 1.0
        Pl[5,0,0] = 1.0
        Pl[5,1,0] = 1.0
        Pl[5,2,0] = 1.0
        Pl[5,3,0] = 1.0
        Pl[6,0,6] = 1.0
        Pl[6,1,6] = 0.5
        Pl[6,1,16] = 0.5
        Pl[6,2,5] = 1.0
        Pl[6,3,7] = 1.0
        Pl[7,0,7] = 1.0
        Pl[7,1,17] = 1.0
        Pl[7,2,6] = 1.0
        Pl[7,3,8] = 1.0
        Pl[8,0,8] = 1.0
        Pl[8,1,18] = 1.0
        Pl[8,2,7] = 1.0
        Pl[8,3,9] = 1.0
        Pl[9,0,9] = 1.0
        Pl[9,1,19] = 1.0
        Pl[9,2,8] = 0.5
        Pl[9,2,9] = 0.5
        Pl[9,3,9] = 1.0
        Pl[10,0,0] = 1.0
        Pl[10,1,20] = 1.0
        Pl[10,2,10] = 1.0
        Pl[10,3,10] = 0.5
        Pl[10,3,11] = 0.5
        Pl[11,0,1] = 1.0
        Pl[11,1,21] = 1.0
        Pl[11,2,10] = 1.0
        Pl[11,3,12] = 1.0
        Pl[12,0,2] = 1.0
        Pl[12,1,12] = 0.5
        Pl[12,1,22] = 0.5
        Pl[12,2,11] = 1.0
        Pl[12,3,13] = 1.0
        Pl[13,0,3] = 1.0
        Pl[13,1,13] = 0.5
        Pl[13,1,23] = 0.5
        Pl[13,2,12] = 1.0
        Pl[13,3,13] = 0.5
        Pl[13,3,14] = 0.5
        Pl[14,0,4] = 1.0
        Pl[14,1,24] = 1.0
        Pl[14,2,13] = 1.0
        Pl[14,3,15] = 1.0
        Pl[15,0,5] = 1.0
        Pl[15,1,25] = 1.0
        Pl[15,2,14] = 1.0
        Pl[15,3,16] = 1.0
        Pl[16,0,6] = 1.0
        Pl[16,1,26] = 1.0
        Pl[16,2,15] = 1.0
        Pl[16,3,17] = 1.0
        Pl[17,0,7] = 0.5
        Pl[17,0,17] = 0.5
        Pl[17,1,17] = 0.5
        Pl[17,1,27] = 0.5
        Pl[17,2,16] = 0.5
        Pl[17,2,17] = 0.5
        Pl[17,3,18] = 1.0
        Pl[18,0,8] = 1.0
        Pl[18,1,28] = 1.0
        Pl[18,2,17] = 1.0
        Pl[18,3,19] = 1.0
        Pl[19,0,9] = 0.5
        Pl[19,0,19] = 0.5
        Pl[19,1,29] = 1.0
        Pl[19,2,18] = 1.0
        Pl[19,3,19] = 1.0
        Pl[20,0,10] = 0.5
        Pl[20,0,20] = 0.5
        Pl[20,1,30] = 1.0
        Pl[20,2,20] = 1.0
        Pl[20,3,21] = 1.0
        Pl[21,0,11] = 1.0
        Pl[21,1,31] = 1.0
        Pl[21,2,20] = 1.0
        Pl[21,3,22] = 1.0
        Pl[22,0,12] = 1.0
        Pl[22,1,32] = 1.0
        Pl[22,2,21] = 1.0
        Pl[22,3,23] = 1.0
        Pl[23,0,13] = 1.0
        Pl[23,1,33] = 1.0
        Pl[23,2,22] = 0.5
        Pl[23,2,23] = 0.5
        Pl[23,3,24] = 1.0
        Pl[24,0,0] = 1.0
        Pl[24,1,0] = 1.0
        Pl[24,2,0] = 1.0
        Pl[24,3,0] = 1.0
        Pl[25,0,15] = 1.0
        Pl[25,1,35] = 1.0
        Pl[25,2,24] = 1.0
        Pl[25,3,26] = 1.0
        Pl[26,0,16] = 1.0
        Pl[26,1,36] = 1.0
        Pl[26,2,25] = 1.0
        Pl[26,3,27] = 1.0
        Pl[27,0,17] = 1.0
        Pl[27,1,37] = 1.0
        Pl[27,2,26] = 1.0
        Pl[27,3,27] = 0.5
        Pl[27,3,28] = 0.5
        Pl[28,0,0] = 1.0
        Pl[28,1,0] = 1.0
        Pl[28,2,0] = 1.0
        Pl[28,3,0] = 1.0
        Pl[29,0,19] = 0.5
        Pl[29,0,29] = 0.5
        Pl[29,1,39] = 1.0
        Pl[29,2,28] = 1.0
        Pl[29,3,29] = 1.0
        Pl[30,0,20] = 1.0
        Pl[30,1,40] = 1.0
        Pl[30,2,30] = 1.0
        Pl[30,3,31] = 1.0
        Pl[31,0,21] = 0.5
        Pl[31,0,31] = 0.5
        Pl[31,1,41] = 1.0
        Pl[31,2,30] = 1.0
        Pl[31,3,32] = 1.0
        Pl[32,0,22] = 1.0
        Pl[32,1,42] = 1.0
        Pl[32,2,31] = 1.0
        Pl[32,3,32] = 0.5
        Pl[32,3,33] = 0.5
        Pl[33,0,23] = 0.5
        Pl[33,0,33] = 0.5
        Pl[33,1,43] = 1.0
        Pl[33,2,32] = 1.0
        Pl[33,3,34] = 1.0
        Pl[34,0,24] = 1.0
        Pl[34,1,44] = 1.0
        Pl[34,2,33] = 1.0
        Pl[34,3,34] = 0.5
        Pl[34,3,35] = 0.5
        Pl[35,0,25] = 1.0
        Pl[35,1,45] = 1.0
        Pl[35,2,34] = 0.5
        Pl[35,2,35] = 0.5
        Pl[35,3,36] = 1.0
        Pl[36,0,26] = 1.0
        Pl[36,1,46] = 1.0
        Pl[36,2,35] = 1.0
        Pl[36,3,37] = 1.0
        Pl[37,0,0] = 1.0
        Pl[37,1,0] = 1.0
        Pl[37,2,0] = 1.0
        Pl[37,3,0] = 1.0
        Pl[38,0,28] = 1.0
        Pl[38,1,38] = 0.5
        Pl[38,1,48] = 0.5
        Pl[38,2,37] = 1.0
        Pl[38,3,39] = 1.0
        Pl[39,0,29] = 1.0
        Pl[39,1,49] = 1.0
        Pl[39,2,38] = 1.0
        Pl[39,3,39] = 1.0
        Pl[40,0,0] = 1.0
        Pl[40,1,0] = 1.0
        Pl[40,2,0] = 1.0
        Pl[40,3,0] = 1.0
        Pl[41,0,31] = 0.5
        Pl[41,0,41] = 0.5
        Pl[41,1,51] = 1.0
        Pl[41,2,40] = 1.0
        Pl[41,3,42] = 1.0
        Pl[42,0,0] = 1.0
        Pl[42,1,0] = 1.0
        Pl[42,2,0] = 1.0
        Pl[42,3,0] = 1.0
        Pl[43,0,33] = 1.0
        Pl[43,1,53] = 1.0
        Pl[43,2,42] = 1.0
        Pl[43,3,44] = 1.0
        Pl[44,0,0] = 1.0
        Pl[44,1,0] = 1.0
        Pl[44,2,0] = 1.0
        Pl[44,3,0] = 1.0
        Pl[45,0,35] = 0.5
        Pl[45,0,45] = 0.5
        Pl[45,1,55] = 1.0
        Pl[45,2,44] = 1.0
        Pl[45,3,46] = 1.0
        Pl[46,0,36] = 0.5
        Pl[46,0,46] = 0.5
        Pl[46,1,56] = 1.0
        Pl[46,2,45] = 1.0
        Pl[46,3,47] = 1.0
        Pl[47,0,37] = 1.0
        Pl[47,1,57] = 1.0
        Pl[47,2,46] = 1.0
        Pl[47,3,48] = 1.0
        Pl[48,0,0] = 1.0
        Pl[48,1,0] = 1.0
        Pl[48,2,0] = 1.0
        Pl[48,3,0] = 1.0
        Pl[49,0,39] = 0.5
        Pl[49,0,49] = 0.5
        Pl[49,1,59] = 1.0
        Pl[49,2,48] = 1.0
        Pl[49,3,49] = 1.0
        Pl[50,0,0] = 1.0
        Pl[50,1,0] = 1.0
        Pl[50,2,0] = 1.0
        Pl[50,3,0] = 1.0
        Pl[51,0,41] = 1.0
        Pl[51,1,61] = 1.0
        Pl[51,2,50] = 1.0
        Pl[51,3,52] = 1.0
        Pl[52,0,42] = 1.0
        Pl[52,1,62] = 1.0
        Pl[52,2,51] = 1.0
        Pl[52,3,53] = 1.0
        Pl[53,0,0] = 1.0
        Pl[53,1,0] = 1.0
        Pl[53,2,0] = 1.0
        Pl[53,3,0] = 1.0
        Pl[54,0,0] = 1.0
        Pl[54,1,0] = 1.0
        Pl[54,2,0] = 1.0
        Pl[54,3,0] = 1.0
        Pl[55,0,0] = 1.0
        Pl[55,1,0] = 1.0
        Pl[55,2,0] = 1.0
        Pl[55,3,0] = 1.0
        Pl[56,0,46] = 0.5
        Pl[56,0,56] = 0.5
        Pl[56,1,66] = 1.0
        Pl[56,2,55] = 1.0
        Pl[56,3,57] = 1.0
        Pl[57,0,47] = 1.0
        Pl[57,1,67] = 1.0
        Pl[57,2,56] = 1.0
        Pl[57,3,58] = 1.0
        Pl[58,0,48] = 1.0
        Pl[58,1,68] = 1.0
        Pl[58,2,57] = 1.0
        Pl[58,3,59] = 1.0
        Pl[59,0,49] = 1.0
        Pl[59,1,69] = 1.0
        Pl[59,2,58] = 1.0
        Pl[59,3,59] = 1.0
        Pl[60,0,50] = 1.0
        Pl[60,1,70] = 1.0
        Pl[60,2,60] = 1.0
        Pl[60,3,61] = 1.0
        Pl[61,0,51] = 1.0
        Pl[61,1,71] = 1.0
        Pl[61,2,60] = 1.0
        Pl[61,3,62] = 1.0
        Pl[62,0,52] = 1.0
        Pl[62,1,72] = 1.0
        Pl[62,2,61] = 1.0
        Pl[62,3,63] = 1.0
        Pl[63,0,53] = 1.0
        Pl[63,1,73] = 1.0
        Pl[63,2,62] = 0.5
        Pl[63,2,63] = 0.5
        Pl[63,3,64] = 1.0
        Pl[64,0,0] = 1.0
        Pl[64,1,0] = 1.0
        Pl[64,2,0] = 1.0
        Pl[64,3,0] = 1.0
        Pl[65,0,55] = 1.0
        Pl[65,1,75] = 1.0
        Pl[65,2,64] = 1.0
        Pl[65,3,65] = 0.5
        Pl[65,3,66] = 0.5
        Pl[66,0,56] = 1.0
        Pl[66,1,66] = 0.5
        Pl[66,1,76] = 0.5
        Pl[66,2,65] = 1.0
        Pl[66,3,67] = 1.0
        Pl[67,0,57] = 1.0
        Pl[67,1,77] = 1.0
        Pl[67,2,66] = 1.0
        Pl[67,3,68] = 1.0
        Pl[68,0,58] = 1.0
        Pl[68,1,68] = 0.5
        Pl[68,1,78] = 0.5
        Pl[68,2,67] = 0.5
        Pl[68,2,68] = 0.5
        Pl[68,3,69] = 1.0
        Pl[69,0,0] = 1.0
        Pl[69,1,0] = 1.0
        Pl[69,2,0] = 1.0
        Pl[69,3,0] = 1.0
        Pl[70,0,0] = 1.0
        Pl[70,1,0] = 1.0
        Pl[70,2,0] = 1.0
        Pl[70,3,0] = 1.0
        Pl[71,0,0] = 1.0
        Pl[71,1,0] = 1.0
        Pl[71,2,0] = 1.0
        Pl[71,3,0] = 1.0
        Pl[72,0,62] = 1.0
        Pl[72,1,82] = 1.0
        Pl[72,2,71] = 0.5
        Pl[72,2,72] = 0.5
        Pl[72,3,73] = 1.0
        Pl[73,0,63] = 1.0
        Pl[73,1,83] = 1.0
        Pl[73,2,72] = 1.0
        Pl[73,3,74] = 1.0
        Pl[74,0,0] = 1.0
        Pl[74,1,0] = 1.0
        Pl[74,2,0] = 1.0
        Pl[74,3,0] = 1.0
        Pl[75,0,65] = 1.0
        Pl[75,1,85] = 1.0
        Pl[75,2,74] = 1.0
        Pl[75,3,75] = 0.5
        Pl[75,3,76] = 0.5
        Pl[76,0,66] = 0.5
        Pl[76,0,76] = 0.5
        Pl[76,1,86] = 1.0
        Pl[76,2,75] = 1.0
        Pl[76,3,77] = 1.0
        Pl[77,0,0] = 1.0
        Pl[77,1,0] = 1.0
        Pl[77,2,0] = 1.0
        Pl[77,3,0] = 1.0
        Pl[78,0,68] = 0.5
        Pl[78,0,78] = 0.5
        Pl[78,1,88] = 1.0
        Pl[78,2,77] = 1.0
        Pl[78,3,78] = 0.5
        Pl[78,3,79] = 0.5
        Pl[79,0,69] = 1.0
        Pl[79,1,79] = 0.5
        Pl[79,1,89] = 0.5
        Pl[79,2,78] = 1.0
        Pl[79,3,79] = 1.0
        Pl[80,0,70] = 1.0
        Pl[80,1,90] = 1.0
        Pl[80,2,80] = 1.0
        Pl[80,3,81] = 1.0
        Pl[81,0,71] = 0.5
        Pl[81,0,81] = 0.5
        Pl[81,1,91] = 1.0
        Pl[81,2,80] = 1.0
        Pl[81,3,82] = 1.0
        Pl[82,0,72] = 1.0
        Pl[82,1,92] = 1.0
        Pl[82,2,81] = 0.5
        Pl[82,2,82] = 0.5
        Pl[82,3,83] = 1.0
        Pl[83,0,73] = 1.0
        Pl[83,1,93] = 1.0
        Pl[83,2,82] = 1.0
        Pl[83,3,83] = 0.5
        Pl[83,3,84] = 0.5
        Pl[84,0,74] = 1.0
        Pl[84,1,94] = 1.0
        Pl[84,2,83] = 0.5
        Pl[84,2,84] = 0.5
        Pl[84,3,85] = 1.0
        Pl[85,0,75] = 0.5
        Pl[85,0,85] = 0.5
        Pl[85,1,95] = 1.0
        Pl[85,2,84] = 1.0
        Pl[85,3,86] = 1.0
        Pl[86,0,76] = 0.5
        Pl[86,0,86] = 0.5
        Pl[86,1,96] = 1.0
        Pl[86,2,85] = 0.5
        Pl[86,2,86] = 0.5
        Pl[86,3,86] = 0.5
        Pl[86,3,87] = 0.5
        Pl[87,0,77] = 1.0
        Pl[87,1,97] = 1.0
        Pl[87,2,86] = 0.5
        Pl[87,2,87] = 0.5
        Pl[87,3,88] = 1.0
        Pl[88,0,78] = 1.0
        Pl[88,1,98] = 1.0
        Pl[88,2,87] = 1.0
        Pl[88,3,89] = 1.0
        Pl[89,0,0] = 1.0
        Pl[89,1,0] = 1.0
        Pl[89,2,0] = 1.0
        Pl[89,3,0] = 1.0
        Pl[90,0,80] = 1.0
        Pl[90,1,90] = 1.0
        Pl[90,2,90] = 1.0
        Pl[90,3,91] = 1.0
        Pl[91,0,81] = 1.0
        Pl[91,1,91] = 1.0
        Pl[91,2,90] = 1.0
        Pl[91,3,91] = 0.5
        Pl[91,3,92] = 0.5
        Pl[92,0,82] = 1.0
        Pl[92,1,92] = 1.0
        Pl[92,2,91] = 1.0
        Pl[92,3,93] = 1.0
        Pl[93,0,83] = 1.0
        Pl[93,1,93] = 1.0
        Pl[93,2,92] = 1.0
        Pl[93,3,93] = 0.5
        Pl[93,3,94] = 0.5
        Pl[94,0,84] = 1.0
        Pl[94,1,94] = 1.0
        Pl[94,2,93] = 1.0
        Pl[94,3,95] = 1.0
        Pl[95,0,85] = 1.0
        Pl[95,1,95] = 1.0
        Pl[95,2,94] = 0.5
        Pl[95,2,95] = 0.5
        Pl[95,3,96] = 1.0
        Pl[96,0,86] = 1.0
        Pl[96,1,96] = 1.0
        Pl[96,2,95] = 1.0
        Pl[96,3,97] = 1.0
        Pl[97,0,0] = 1.0
        Pl[97,1,0] = 1.0
        Pl[97,2,0] = 1.0
        Pl[97,3,0] = 1.0
        Pl[98,0,88] = 0.5
        Pl[98,0,98] = 0.5
        Pl[98,1,98] = 1.0
        Pl[98,2,97] = 1.0
        Pl[98,3,99] = 1.0

        ############################################################################################################

        Rl = np.zeros((100,4))
        Rl[2,0] = -100.0
        Rl[2,1] = -100.0
        Rl[2,2] = -100.0
        Rl[2,3] = -100.0
        Rl[5,0] = -100.0
        Rl[5,1] = -100.0
        Rl[5,2] = -100.0
        Rl[5,3] = -100.0
        Rl[7,0] = -10.0
        Rl[7,1] = -10.0
        Rl[7,2] = -10.0
        Rl[7,3] = -10.0
        Rl[9,0] = -10.0
        Rl[9,1] = -10.0
        Rl[9,2] = -10.0
        Rl[9,3] = -10.0
        Rl[11,0] = -10.0
        Rl[11,1] = -10.0
        Rl[11,2] = -10.0
        Rl[11,3] = -10.0
        Rl[12,0] = -10.0
        Rl[12,1] = -10.0
        Rl[12,2] = -10.0
        Rl[12,3] = -10.0
        Rl[21,0] = -10.0
        Rl[21,1] = -10.0
        Rl[21,2] = -10.0
        Rl[21,3] = -10.0
        Rl[22,0] = -10.0
        Rl[22,1] = -10.0
        Rl[22,2] = -10.0
        Rl[22,3] = -10.0
        Rl[23,0] = -10.0
        Rl[23,1] = -10.0
        Rl[23,2] = -10.0
        Rl[23,3] = -10.0
        Rl[24,0] = -100.0
        Rl[24,1] = -100.0
        Rl[24,2] = -100.0
        Rl[24,3] = -100.0
        Rl[26,0] = -10.0
        Rl[26,1] = -10.0
        Rl[26,2] = -10.0
        Rl[26,3] = -10.0
        Rl[27,0] = -10.0
        Rl[27,1] = -10.0
        Rl[27,2] = -10.0
        Rl[27,3] = -10.0
        Rl[28,0] = -100.0
        Rl[28,1] = -100.0
        Rl[28,2] = -100.0
        Rl[28,3] = -100.0
        Rl[33,0] = -10.0
        Rl[33,1] = -10.0
        Rl[33,2] = -10.0
        Rl[33,3] = -10.0
        Rl[35,0] = -10.0
        Rl[35,1] = -10.0
        Rl[35,2] = -10.0
        Rl[35,3] = -10.0
        Rl[37,0] = -100.0
        Rl[37,1] = -100.0
        Rl[37,2] = -100.0
        Rl[37,3] = -100.0
        Rl[38,0] = -10.0
        Rl[38,1] = -10.0
        Rl[38,2] = -10.0
        Rl[38,3] = -10.0
        Rl[40,0] = -100.0
        Rl[40,1] = -100.0
        Rl[40,2] = -100.0
        Rl[40,3] = -100.0
        Rl[41,0] = -10.0
        Rl[41,1] = -10.0
        Rl[41,2] = -10.0
        Rl[41,3] = -10.0
        Rl[42,0] = -100.0
        Rl[42,1] = -100.0
        Rl[42,2] = -100.0
        Rl[42,3] = -100.0
        Rl[43,0] = -10.0
        Rl[43,1] = -10.0
        Rl[43,2] = -10.0
        Rl[43,3] = -10.0
        Rl[44,0] = -100.0
        Rl[44,1] = -100.0
        Rl[44,2] = -100.0
        Rl[44,3] = -100.0
        Rl[45,0] = -10.0
        Rl[45,1] = -10.0
        Rl[45,2] = -10.0
        Rl[45,3] = -10.0
        Rl[46,0] = -10.0
        Rl[46,1] = -10.0
        Rl[46,2] = -10.0
        Rl[46,3] = -10.0
        Rl[48,0] = -100.0
        Rl[48,1] = -100.0
        Rl[48,2] = -100.0
        Rl[48,3] = -100.0
        Rl[50,0] = -100.0
        Rl[50,1] = -100.0
        Rl[50,2] = -100.0
        Rl[50,3] = -100.0
        Rl[53,0] = -100.0
        Rl[53,1] = -100.0
        Rl[53,2] = -100.0
        Rl[53,3] = -100.0
        Rl[54,0] = -100.0
        Rl[54,1] = -100.0
        Rl[54,2] = -100.0
        Rl[54,3] = -100.0
        Rl[55,0] = -100.0
        Rl[55,1] = -100.0
        Rl[55,2] = -100.0
        Rl[55,3] = -100.0
        Rl[63,0] = -10.0
        Rl[63,1] = -10.0
        Rl[63,2] = -10.0
        Rl[63,3] = -10.0
        Rl[64,0] = -100.0
        Rl[64,1] = -100.0
        Rl[64,2] = -100.0
        Rl[64,3] = -100.0
        Rl[66,0] = -10.0
        Rl[66,1] = -10.0
        Rl[66,2] = -10.0
        Rl[66,3] = -10.0
        Rl[67,0] = -10.0
        Rl[67,1] = -10.0
        Rl[67,2] = -10.0
        Rl[67,3] = -10.0
        Rl[69,0] = -100.0
        Rl[69,1] = -100.0
        Rl[69,2] = -100.0
        Rl[69,3] = -100.0
        Rl[70,0] = -100.0
        Rl[70,1] = -100.0
        Rl[70,2] = -100.0
        Rl[70,3] = -100.0
        Rl[71,0] = -100.0
        Rl[71,1] = -100.0
        Rl[71,2] = -100.0
        Rl[71,3] = -100.0
        Rl[72,0] = -10.0
        Rl[72,1] = -10.0
        Rl[72,2] = -10.0
        Rl[72,3] = -10.0
        Rl[73,0] = -10.0
        Rl[73,1] = -10.0
        Rl[73,2] = -10.0
        Rl[73,3] = -10.0
        Rl[74,0] = -100.0
        Rl[74,1] = -100.0
        Rl[74,2] = -100.0
        Rl[74,3] = -100.0
        Rl[75,0] = -10.0
        Rl[75,1] = -10.0
        Rl[75,2] = -10.0
        Rl[75,3] = -10.0
        Rl[76,0] = -10.0
        Rl[76,1] = -10.0
        Rl[76,2] = -10.0
        Rl[76,3] = -10.0
        Rl[77,0] = -100.0
        Rl[77,1] = -100.0
        Rl[77,2] = -100.0
        Rl[77,3] = -100.0
        Rl[78,0] = -10.0
        Rl[78,1] = -10.0
        Rl[78,2] = -10.0
        Rl[78,3] = -10.0
        Rl[81,0] = -10.0
        Rl[81,1] = -10.0
        Rl[81,2] = -10.0
        Rl[81,3] = -10.0
        Rl[86,0] = -10.0
        Rl[86,1] = -10.0
        Rl[86,2] = -10.0
        Rl[86,3] = -10.0
        Rl[87,0] = -10.0
        Rl[87,1] = -10.0
        Rl[87,2] = -10.0
        Rl[87,3] = -10.0
        Rl[89,0] = -100.0
        Rl[89,1] = -100.0
        Rl[89,2] = -100.0
        Rl[89,3] = -100.0
        Rl[96,0] = -10.0
        Rl[96,1] = -10.0
        Rl[96,2] = -10.0
        Rl[96,3] = -10.0
        Rl[97,0] = -100.0
        Rl[97,1] = -100.0
        Rl[97,2] = -100.0
        Rl[97,3] = -100.0
        Rl[99,0] = 750.0
        Rl[99,1] = 750.0
        Rl[99,2] = 750.0
        Rl[99,3] = 750.0
        

        self.expl = Explanations(100,Pl,Rl,4,0.90,[99])


    def test1(self):
        self.expl.single_get_state_explanation_distribution(0,labels=['Action UP','Action DOWN','Action LEFT','Action RIGHT'])

    def test2(self):
        self.expl.single_get_state_explanation_distribution(0,change_Pl=False,labels=['Action UP','Action DOWN','Action LEFT','Action RIGHT'])

    def test3(self):
        self.expl.c51_state_distribution(0,-750,750,labels=['Action UP','Action DOWN','Action LEFT','Action RIGHT'])

    def test4(self):
        self.expl.c51_state_distribution(0,-750,750, proc_actions=[3])

    def test5(self):
        path = [0,10,20,30,31,41,51,61,62,72,82,83,84,85,86,87,88,98,99]
        pathactions = [1,1,1,3,1,1,1,3,1,1,3,3,3,3,3,3,1,3]
        self.expl.get_path_distribution(path,pathactions)

    def test6(self):
        path = [0,10,20,30,31,41,51,61,62,72,82,83,84,85,86,87,88,98,99]
        pathactions = [1,1,1,3,1,1,1,3,1,1,3,3,3,3,3,3,1,3]
        bif = self.expl.get_path_bifurcations_outside_path(path,pathactions)

    def test7(self):
        pth = self.expl.get_optimal_path_from_state(0)
        print(pth)

    def test8(self):
        path = [0,10,20,30,31,41,51,61,62,72,82,83,84,85,86,87,88,98,99]
        pathactions = [1,1,1,3,1,1,1,3,1,1,3,3,3,3,3,3,1,3]
        self.expl.get_path_security_distribution(path,pathactions)

    def test9(self):
        self.expl.single_get_state_explanation_distribution(0,proc_actions=[1,3])
    
    def test10(self):
        path = [0,10,20,30,31,41,51,61,62,72,82,83,84,85,86,87,88,98,99]
        pathactions = [1,1,1,3,1,1,1,3,1,1,3,3,3,3,3,3,1,3]
        self.expl.get_path_security_distribution(path,pathactions,epsilon=0.5,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False)

# if __name__ == "__main__":
#     ex1 = example1()
#     ex2 = example2()
#     ex3 = example3()
#     ex4 = example4()

    
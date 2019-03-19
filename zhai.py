from aman import filter_input1
from aman import filter_input2
from aman import cos_matrix
from encoder import Encoder


myenc = Encoder()

directory = "/home/yerlan/HackNU/images/1/input"
yer = {1: ["Aman", "haah"], 2: ["Yerla", "Salam"]}
given = filter_input1(directory)
zhanik = filter_input2(directory)



#Printing Results without Zhanibek
print("Without Zhanibek")

for i in range(len(yer)):
    cos = cos_matrix(yer[i+1], given)
    print("For Image #" + str(i+1))
    print("\tmean: " + str(cos[0]))
    print("\tminimum: " + str(cos[1]))






#Printing Results with Zhanibek
print("With Zhanibek")

for i in range(len(yer)):
    cos = cos_matrix(yer[i+1], zhanik)
    print("For Image #" + str(i+1))
    print("\tmean: " + str(cos[0]))
    print("\tminimum: " + str(cos[1]))
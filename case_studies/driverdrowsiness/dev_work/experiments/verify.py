def main():
    f1_name = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/case_studies/driverdrowsiness/dev_work/experiments/output9.txt"
    f2_name = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/case_studies/driverdrowsiness/dev_work/experiments/Classification_Results2.txt"
    file_1 = []
    file_2 = []

    with open(f1_name, "r") as f:
        x = f.readlines()
        x = [item.strip("\n") for item in x]
        x = [item.split(" ") for item in x]
        file_1 = x.copy()
    
    with open(f2_name, "r") as f:
        x = f.readlines()
        x = [item.strip("\n") for item in x]
        x = [item.split(" ") for item in x]
        file_2 = x.copy()

    n = 0
    if len(file_1) > len(file_2):
        n = len(file_2)
    else: n = len(file_1)

    count = 0
    for i in range(n):
        if not file_1[i][2] == file_2[i][2]:
            print("inference {}: py = {} de = {}".format(file_1[i][0],
                                                         file_1[i][2], 
                                                         file_2[i][2]))
            count += 1
    percent = count / n
    print("percent difference between inferences: {}".format(percent))

if __name__ == "__main__":
    main()
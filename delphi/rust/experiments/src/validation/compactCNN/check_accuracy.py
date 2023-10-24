def main():
    file = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation/Classification_Results2.txt"
    x = []
    with open(file, 'r') as f:
        x = f.readlines()
        for i in range(len(x)):
            x[i] = x[i].split()
            for j in range(len(x[i])):
                x[i][j] = int(x[i][j])
    unique = dict()
    for i in range(len(x)):
        if not x[i][0] in unique:
            unique[x[i][0]]=(x[i])
    correct = 0
    total = 0
    for key in unique:
        if unique[key][1] == unique[key][2]:
            correct += 1
        total += 1
    
    print(f"correct  = {correct}")
    print(f"total    = {total}")
    print(f"accuracy = {correct / total}")

if __name__ == "__main__":
    main()


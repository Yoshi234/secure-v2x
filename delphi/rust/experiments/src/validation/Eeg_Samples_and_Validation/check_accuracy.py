if __name__ == "__main__":
    x = []
    with open("classification_results.txt", 'r') as f:
        x = f.readlines()
        for i in range(len(x)):
            x[i] = x[i].split()
            for j in range(len(x[i])):
                x[i][j] = int(x[i][j])
    correct = 0
    total = 0
    for i in range(len(x)):
        if x[i][1] == x[i][2]:
            correct += 1
        total += 1
    
    print(f"correct  = {correct}")
    print(f"total    = {total}")
    print(f"accuracy = {correct / total}")


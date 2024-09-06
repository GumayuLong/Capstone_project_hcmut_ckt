# Mảng A và B
A = [1, 2, 3, 4, 5, 6]
B = [7, 8, 9, 10]
arr = []
# Dùng vòng lặp để trừ từng phần tử của mảng B từ phần tử tương ứng của mảng A
for i in range(len(A)):
    for j in range(len(B)):
        if(j == i):
            result = abs(A[i] - B[j])
            print(f"A[{i}] - B[{j}] = {result}")
            arr.append(result)
print(arr)
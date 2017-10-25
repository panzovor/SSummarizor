def generate_options(len, num, start=0):
    res = []
    if num == 1:
        for var in range(start, len):
            res.append([var])
        return res
    for i in range(start, len - num + 1):
        for var in generate_options(len, num - 1, i + 1):
            res.append([i] + var)
    return res


tmp = generate_options(10,4,1)

for var in tmp:
    print(var)
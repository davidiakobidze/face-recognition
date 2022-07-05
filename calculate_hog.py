if __name__ == '__main__':
    A = [
        [0, 89, 89, 89, 89, 89, 89, 0],
        [180, 179, 8, 171, 165, 70, 97, 0],
        [180, 175, 176, 157, 139, 172, 89, 180],
        [180, 168, 164, 141, 123, 117, 114, 0],
        [180, 120, 109, 111, 105, 103, 90, 0],
        [0, 86, 101, 87, 89, 95, 122, 180],
        [0, 23, 121, 89, 80, 83, 73, 0],
        [0, 89, 90, 90, 90, 90, 90, 0]
    ]

    M = [
        [0, 22, 96, 10, 32, 158, 80, 0],
        [478, 448, 380, 286, 162, 117, 78, 12],
        [514, 512, 468, 401, 303, 72, 78, 2],
        [288, 280, 266, 327, 393, 186, 57, 52],
        [110, 131, 78, 174, 386, 293, 8, 32],
        [34, 129, 78, 142, 351, 309, 60, 22],
        [124, 42, 93, 211, 334, 278, 68, 24],
        [0, 68, 78, 264, 328, 258, 70, 0]
    ]

    bin_values = [
        0,
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        160,
    ]

    result = {
        0: 0,
        20: 0,
        40: 0,
        60: 0,
        80: 0,
        100: 0,
        120: 0,
        140: 0,
        160: 0,
    }


    def get_bin_index(n):
        if n > 160:
            return 8, 0, 160, 0
        for index, value in enumerate(bin_values):
            if n < value:
                return index - 1, index, bin_values[index - 1], value


    for i, a_row in enumerate(A):
        for j, a in enumerate(a_row):
            if a in bin_values:
                result[a] += a
                continue
            bin_index_left, bin_index_right, left, right = get_bin_index(a)
            _result_right = (a - bin_values[bin_index_left]) / 20 * M[i][j]
            ll = bin_values[bin_index_right] if a < 160 else 180
            _result_left = (ll - a) / 20 * M[i][j]
            result[left] += _result_left
            result[right] += _result_right

    dd = 0
    for a_r in M:
        for a in a_r:
            dd+=a

    print(dd)
    print(sum(result.values()))
    print(result)

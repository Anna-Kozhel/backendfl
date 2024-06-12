import numpy as np


class DivisionIntoLexemes:
    def __init__(self, matrixImage):
        self.matrixImage = matrixImage
        self.net, self.overlay, self.corrected_image, self.lexemsList = self.predictText()

    def isPoint(self, img, x, y):
        if img[x + 1][y] == img[x + 2][y] == img[x + 3][y] \
                or img[x - 1][y] == img[x - 2][y] == img[x - 3][y] \
                or img[x][y - 1] == img[x][y - 2] == img[x][y - 3] \
                or img[x][y + 1] == img[x][y + 2] == img[x][y + 3] \
                or img[x][y + 1] == img[x][y - 1] == img[x + 1][y] == img[x - 1][y]:
            return True
        else:
            return False

    def pointsNearby(self, eroded_image, minWidth, minHeight):
        cellHeight, cellWidth = 0, 0
        for row_idx, row in enumerate(eroded_image):
            for col_idx, point in enumerate(row):
                if point == 0:
                    if self.isPoint(eroded_image, row_idx, col_idx):
                        length, flag, spec = 0, 0, 0
                        while True:
                            if row[col_idx] == 0 and flag == 0:
                                length += 1.5
                            else:
                                flag = 1
                                spec += 1
                                round(length)
                                if (length <= 0 and row[col_idx] == 255) or (col_idx >= len(row)):
                                    flag, spec, length = 0, 0, 0
                                    break
                                elif row[col_idx] == 0:
                                    if self.isPoint(eroded_image, row_idx, col_idx):
                                        cellWidth = spec + 2 * (minWidth - 1) + 1
                                        break
                                length -= 1
                            col_idx += 1
                            if col_idx >= len(row): break

        for col_idx in range(eroded_image.shape[1]):
            for row_idx in range(eroded_image.shape[0]):
                if eroded_image[row_idx][col_idx] == 0:
                    if self.isPoint(eroded_image, row_idx, col_idx):
                        length, flag, spec = 0, 0, 0
                        while True:
                            if eroded_image[row_idx][col_idx] == 0 and flag == 0:
                                length += 1
                            else:
                                flag = 1
                                spec += 1
                                round(length)
                                if (length <= 0 and eroded_image[row_idx][col_idx] == 255) or (
                                        row_idx >= eroded_image.shape[0]):
                                    length, flag, spec = 0, 0, 0
                                    break
                                elif eroded_image[row_idx][col_idx] == 0:
                                    if self.isPoint(eroded_image, row_idx, col_idx):
                                        cellHeight = spec * 2 + (minHeight - 1) * 3 + 1
                                        break
                                length -= 1
                            row_idx += 1
                            if row_idx >= eroded_image.shape[0]:
                                break
        return cellWidth, cellHeight

    def positioning(self, image, overlay):
        images, overlays, dots = [image], [overlay], [3, 2]
        for dot in dots:
            first, second = [], []
            for l in range(len(overlays)):
                images[l], overlays[l] = images[l].T, overlays[l].T
                flag, borders, index, line = 0, [0, 0], 0, 0
                for i in range(overlays[l].shape[0]):
                    if (overlays[l][i] == 255).all():
                        continue
                    else:
                        index, line = i, overlays[l][i]
                        break
                length, j = len(line), 0
                while j < length:
                    if j < length - 1:
                        if line[j + 1] == 0 and line[j] == 255 and flag == 0:
                            if borders[0] == 0: borders[0] = j + 1
                        if line[j] == 0 and line[j + 1] == 255:
                            flag += 1
                            j += 1
                            continue
                    if flag == dot and line[j] == 255: borders[1] += 1
                    if flag == dot and (line[j] == 0 or j == len(line) - 1):
                        borders = [borders[0], borders[1] / 2]
                        border = round(min(borders))
                        cut = [borders[0] - border, j - (border)]
                        first.append(images[l][:, cut[0]:cut[1]])
                        second.append(overlays[l][:, cut[0]:cut[1]])
                        images[l] = images[l][:, cut[1]:]
                        overlays[l] = overlays[l][:, cut[1]:]
                        line = line[cut[1]:]
                        length = len(line)
                        j, borders, flag = border - 1, [0, 0], 0
                        continue
                    j += 1
            if dot == dots[0]:
                images = first
                overlays = second
        images = first
        overlays = second
        return images

    def net(self, img):
        flag = 0
        netImg = np.ones_like(img)
        for _ in range(2):
            netImg, img = netImg.T, img.T
            for i in range(img.shape[0]):
                if (img[i] == 255).all(): continue
                for j in range(img.shape[1]):
                    if img[i][j] == 0 and flag == 0:
                        netImg[i - 1], flag = [0 for _ in range(img.shape[1])], 1
                    elif flag == 1 and (img[i + 1] == 255).all():
                        netImg[i + 1], flag = [0 for _ in range(img.shape[1])], 0
                        break
                    elif (not (img[i] == 255).all()) and flag == 1:
                        break
        netImg *= 255
        lineSegment, skipAvgArr, dotAvgArr, dotAvg, skips = self.fillGap(img, netImg)
        netImg = self.netGap(netImg, lineSegment, skipAvgArr, skips, dotAvg)
        return netImg

    def fillCells(self, model):
        flag, pointer = 0, 0
        for i in range(model[0].shape[0]):
            if not ((model[0][i] == 255).all()) and pointer != 0: continue
            if (model[0][i] == 255).all():
                pointer = 0
                continue
            for j in range(model[0].shape[1]):
                if j >= model[0].shape[1] - 2:
                    pointer = 1
                    break
                if (model[flag][i][j] == 255 and model[flag][i][j + 1] == 0 and flag == 1): flag = 2
                if flag == 2 or (flag == 0 and model[flag][i][j] == 0 and model[flag][i][j + 1] == 255):
                    if flag == 2:
                        model[0], flag = self.fillPoint(model, i, j + 2), 0
                        continue
                    flag = 1
        return model[0]

    def fillPoint(self, model, x, y):
        for i in range(x, model[0].shape[0]):
            for j in range(y, model[0].shape[1]):
                if model[1][i][j] == 255:
                    model[0][i][j] = 0
                else:
                    break
            if (model[1][i] == 0).all(): break
        return model[0]

    def fillGap(self, pattern, net):
        dotAvgArr, skipAvgArr, black, white = [], [], 0, 0
        line, lineSegment = net[0], []
        for i in range(len(line)):
            if line[i] == 255:
                if black != 0: lineSegment.append(black)
                black, white = 0, white + 1
            if line[i] == 0 or i == len(line) - 1:
                if white != 0: lineSegment.append(white)
                white, black = 0, 1
        whiteSegment = list(filter(lambda x: x != 1, lineSegment))
        for i in range(1, len(whiteSegment), 2):
            skipAvgArr.append(whiteSegment[i - 1] + 2)
            dotAvgArr.append(whiteSegment[i])
        skipAvgArr = skipAvgArr[1:len(skipAvgArr)]
        dotAvg = np.mean(dotAvgArr)
        skipMin = np.min(skipAvgArr)
        skipMax = np.max(skipAvgArr)
        skips = []
        for i in range(0, 10):
            if i * dotAvg + (i + 1) * skipMin <= skipMax: skips.append(i * dotAvg + (i + 1) * skipMin)
        return lineSegment, skipAvgArr, dotAvgArr, dotAvg, skips

    def netGap(self, net, lineSegment, skipAvgArr, skips, dotAvg):
        for i in range(len(skips) - 1):
            for j in range(len(skipAvgArr)):
                if len(skips) == 1:
                    return net
                else:
                    compares = [abs(skipAvgArr[j] - skips[k]) for k in range(len(skips))]
                    comperesMin = min(compares)
                    indexC = compares.index(comperesMin)
                    if indexC == 0: continue
                    length = round((skipAvgArr[j] - indexC * (dotAvg - 2)) / (indexC + 1)) + 1
                    index = sum(lineSegment[:3 + j * 4]) + length
                    net = net.T
                    for k in range(1, indexC + 1):
                        net[index] = [0 for _ in range(len(net[index]))]
                        index += round(dotAvg + 1)
                        net[index] = [0 for _ in range(len(net[index]))]
                        index += length
                    net = net.T
        return net

    def predictText(self):
        n_row = self.matrixImage.shape[0]
        n_col = self.matrixImage.shape[1]
        for k in range(0, 1):
            z = k % 2
            for i in range(0, n_row, 2):
                for j in range(0 + z, n_col, 2):
                    if j < n_col and i < n_row:
                        if self.matrixImage[i][j] == 0:
                            if j + 1 < n_col and i + 1 < n_row:
                                self.matrixImage[i][j + 1] = 0
                                self.matrixImage[i + 1][j] = 0
                                self.matrixImage[i + 1][j + 1] = 0
            for i in reversed(range(0, n_row, 2)):
                for j in reversed(range(0, n_col - z, 2)):
                    if j > 0 and i > 0:
                        if self.matrixImage[i][j] == 0:
                            if j - 1 < n_col and i - 1 > 0:
                                self.matrixImage[i][j - 1] = 0
                                self.matrixImage[i - 1][j] = 0
                                self.matrixImage[i - 1][j - 1] = 0
            for j in range(0, n_col, 2):
                for i in range(0 + z, n_row, 2):
                    if i < n_row and i < n_row:
                        if self.matrixImage[i][j] == 0:
                            if i + 1 < n_row and j - 1 < 0:
                                self.matrixImage[i + 1][j] = 0
                                self.matrixImage[i][j - 1] = 0
                                self.matrixImage[i + 1][j - 1] = 0
            for j in reversed(range(0, n_col, 2)):
                for i in reversed(range(0, n_row - z, 2)):
                    if i > 0 and j > 0:
                        if self.matrixImage[i][j] == 0:
                            if i - 1 > 0 and j + 1 < n_col:
                                self.matrixImage[i - 1][j] = 0
                                self.matrixImage[i][j + 1] = 0
                                self.matrixImage[i - 1][j + 1] = 0

        row, col, height, width = [[]], [[]], [], []
        k, v = 0, 0

        for i in range(n_row):
            for j in range(n_col):
                if self.matrixImage[i][j] == 0 and len(row[k]) == 0:
                    row[k].append(i)
                    break
                elif len(row[k]) != 0 and np.all(self.matrixImage[i] != 0):
                    row[k].append(i)
                    height.append(row[k][1] - row[k][0])
                    row.append([])
                    k += 1
                    break
        row.pop()

        for j in range(n_col):
            for i in range(n_row):
                if self.matrixImage[i][j] == 0 and len(col[v]) == 0:
                    col[v].append(j)
                    break
                elif len(col[v]) != 0 and np.all(self.matrixImage[:, j] != 0):
                    col[v].append(j)
                    width.append(col[v][1] - col[v][0])
                    col.append([])
                    v += 1
                    break

        col.pop()

        minHeight = min(height)
        minWidth = min(width)
        # lock = min(minWidth, minHeight)
        # minWidth = lock
        # minHeight = lock

        epochHeight = [height[i] - minHeight for i in range(k)]
        epochWidth = [width[i] - minWidth for i in range(v)]
        corrected_image = np.array(np.full((n_row, n_col), 255)).astype(np.uint8)

        for n in range(k):
            for i in range(row[n][0], row[n][1] - epochHeight[n]):
                for m in range(v):
                    for j in range(col[m][0], col[m][1] - epochWidth[m]):
                        if np.all(self.matrixImage[row[n][0]:row[n][1], col[m][0]:col[m][1]] == 255):
                            continue
                        else:
                            corrected_image[i][j] = 0
        cellWidth, cellHeight = self.pointsNearby(corrected_image, minWidth, minHeight)
        net = self.net(corrected_image)
        overlay = self.fillCells([corrected_image.copy(), net])
        return net, overlay, corrected_image, self.positioning(corrected_image, overlay)

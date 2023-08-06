import copy
import numpy as np
import cv2

from tubes import Tubes

MAX_HEIGHT = 4


class Game(Tubes):
    def __init__(self, img_path):
        super().__init__(img_path)
        print(self.warped_ipad.shape)
        self.answer = []

    def algorithm(self):
        tubes = self.colors
        tubes = self.tubes2String(tubes)
        for i, t in enumerate(tubes):
            if (i < len(tubes) - 1):
                answer = []
                new_game = copy.copy(tubes)
                cond, answer = self.solve([], new_game, answer)

                # answers are pushed in reverse order, so reverse it to get answer from beginning
                answer.reverse()

                if len(answer) > 0:
                    for ind, a in enumerate(answer):
                        print(a)
                        self.answer.append(a)
                        if self.can_pour(tubes[a[0]], tubes[a[1]], tubes):
                            self.pour(tubes[a[0]], tubes[a[1]])

                    break
                else:
                    first_tube = tubes.pop()
                    tubes.insert(0, first_tube)
                    # print(self.tubes)
        return tubes

    def tubeMovesToPos(self):
        move_pos = []
        for move in self.answer:
            for m in move:
                move_pos.append(self.tube_centers_original[m])
        return move_pos
        # count the number of base colors in tube

    def count_color(self, tube, color):
        count = 0
        for c in tube:
            if c == color:
                count += 1
        return count

    # check if all tubes have been sorted

    def is_solved(self, tubes):
        for t in tubes:
            if (len(t) > 0):
                if (len(t) < MAX_HEIGHT):
                    return False
                if (self.count_color(t, t[0]) != MAX_HEIGHT):
                    return False
        return True

    # check if source tube can pour into destination tube

    def can_pour(self, source, destination, tubes):
        if len(source) == 0 or len(destination) == MAX_HEIGHT:
            return False
        source_colors = self.count_color(source, source[0])

        if source_colors == MAX_HEIGHT:
            return False

        destination_colors = 0
        if len(destination):
            destination_colors = self.count_color(destination, destination[0])

        # removes unnecessary pours to already filled tubes
        if self.count_color(source, source[-1]) + len(destination) > MAX_HEIGHT:
            return False

        better_option_found = False
        # attempting to look at other tubes
        for ind, tube in enumerate(tubes):
            if tube == destination:
                continue
            # check if a different tube is better to move to
            tube_colors = self.count_color(
                tube, tube[0]) if len(tube) > 0 else 0
            if len(tube) != MAX_HEIGHT and len(tube) > 0 and source[-1] == tube[-1] and tube_colors > destination_colors:
                # better option in tubes, so therefore we don't pour here
                better_option_found = True
                break

        if better_option_found:
            return False

        if not len(destination):  # empty empty list
            if source_colors == len(source):
                return False
            return True

        return source[-1] == destination[-1]

    # pour source tube into destination tube

    def pour(self, source, destination):

        # always move one
        top = source.pop()
        destination.append(top)
        while len(source) > 0:
            # look at next and compare
            next = source[len(source) - 1]
            if (next == top):
                destination.append(source.pop())
            else:
                break

    def tubes2String(self, tubes):
        tubeStr = []
        for t in tubes:
            newTube = []
            for c in t:
                newTube.append(str(c))
            tubeStr.append(newTube)
        return tubeStr

    # recursively solve the tubes by storing visited sorts

    def solve(self, visited_sorts, current_sort, answer):
        visited_sorts.append(self.tubes2String(current_sort))

        for i1, t1 in enumerate(current_sort):
            for i2, t2 in enumerate(current_sort):
                if (i1 != i2 and self.can_pour(t1, t2, current_sort)):
                    new_sort = copy.deepcopy(current_sort)
                    if self.can_pour(new_sort[i1], new_sort[i2], new_sort):
                        self.pour(new_sort[i1], new_sort[i2])

                        if (self.is_solved(new_sort)):
                            print('SOLVED')
                            answer.append([i1, i2])
                            return True, answer
                        if (self.tubes2String(new_sort) not in visited_sorts):
                            continue_sort, updated_answer = self.solve(
                                visited_sorts, new_sort, answer)
                            if (continue_sort):
                                answer.append([i1, i2])
                                return True, updated_answer

        return False, answer


game = Game('tubes.jpg')


print(game.algorithm())
print(game.tubeMovesToPos())

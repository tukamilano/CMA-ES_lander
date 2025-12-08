import math

class Board():
    '''
    物理システムと終了条件の記述
    '''
    def __init__(self, ground_height_list, lander_pos, lander_speed, flat_ground_pair):
        self.width = 7000
        self.height = 3000
        self.gravity = 3.711
        self.ground_height_list = ground_height_list
        self.lander_pos = lander_pos
        self.lander_speed = lander_speed
        self.flat_ground_pair = flat_ground_pair
    
    def is_terminate(self):
        if self.lander_pos[1] <= self.ground_height_list[self.lander_pos[0]]: #着地
            if (self.ground_pair[0] <= self.lander_pos[0]) and (self.lander_pos[0] <= self.ground_pair[1]):
                return True, 'success'
            else:
                return True, 'false'
        elif (20 < abs(self.lander_speed[0])) or (40 < abs(self.lander_speed[1])):
            return True, 'false'
        elif (self.lander_pos[0] < 0) or (self.width <= self.lander_pos[0]) or (self.height <= self.lander_pos[1]):
            return True, 'false'
        else:
            return False, None
    
    def update(self, accelerate_power, accelerate_angle):
        A = self.gravity * accelerate_power / 4
        accelerate = (-A * math.sin(accelerate_angle), A * math.cos(accelerate_angle) - self.gravity)
        self.lander_pos[0] += int(self.lander_speed[0] + (accelerate[0] / 2))
        self.lander_pos[1] += int(self.lander_speed[1] + (accelerate[1] / 2))

            
surfaceN = int(input())
land_XY_list = []
for _ in range(surfaceN):
    land_XY_list.append(tuple(map(int, input().split())))

y_list = [0] * 7000
for (x1, y1), (x2, y2) in zip(land_XY_list, land_XY_list[1:]):
    dx = x2 - x1
    dy = y2 - y1
    for x in range(x1, x2 + 1):
        t = (x - x1) / dx
        y = y1 + t * dy
        y_list[x] = int(round(y))

X, Y, hSpeed, vSpeed, fuel, rotate, power = map(int, input().split())

# Board(状態の構成)


    

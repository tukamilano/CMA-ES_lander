import math

class Board():
    '''
    物理システムと終了条件の記述
    '''
    def __init__(self, ground_height_list, lander_pos, lander_speed, flat_ground_pair, init_rotate, init_power, init_fuel):
        self.width = 7000
        self.height = 3000
        self.gravity = 3.711
        self.ground_height_list = ground_height_list
        self.lander_pos = lander_pos
        self.lander_speed = lander_speed
        self.flat_ground_pair = flat_ground_pair
        #未実装
        self.rotate = init_rotate
        self.power = init_power
        self.fuel = init_fuel
    
    def is_terminate(self):
        if self.lander_pos[1] <= self.ground_height_list[self.lander_pos[0]]: #着地
            if (self.flat_ground_pair[0] <= self.lander_pos[0]) and (self.lander_pos[0] <= self.flat_ground_pair[1]):
                return True, self.score(self)
            else:
                return True, self.score(self)
        elif (20 < abs(self.lander_speed[0])) or (40 < abs(self.lander_speed[1])):
            return True, self.score(self)
        elif (self.lander_pos[0] < 0) or (self.width <= self.lander_pos[0]) or (self.height <= self.lander_pos[1]):
            return True, self.score(self)
        else:
            return False, None
        
    def score(self):
        pass
        '''
                // Score is used to order landers by performance
        calculateScore: function(level, hitLandingArea) {
            var currentSpeed = Math.sqrt(Math.pow(this.xspeed, 2) + Math.pow(this.yspeed, 2));

            // 0-100: crashed somewhere, calculate score by distance to landing area
            if (!hitLandingArea) {

                var lastX = this.points[this.points.length-2][0];
                var lastY = this.points[this.points.length-2][1];
                var distance = level.getDistanceToLandingArea(lastX, lastY);

                // Calculate score from distance
                this.score = 100 - (100 * distance / level.max_dist);

                // High speeds are bad, they decrease maneuvrability
                var speedPen = 0.1 * Math.max(currentSpeed - 100, 0);
                this.score -= speedPen;
            }

            // 100-200: crashed into landing area, calculate score by speed above safety
            else if (this.yspeed < -40 || 20 < Math.abs(this.xspeed)) {
                var xPen = 0;
                if (20 < Math.abs(this.xspeed)) {
                    xPen = (Math.abs(this.xspeed) - 20) / 2
                }
                var yPen = 0
                if (this.yspeed < -40) {
                    yPen = (-40 - this.yspeed) / 2
                }
                this.score = 200 - xPen - yPen
                return;
            }

            // 200-300: landed safely, calculate score by fuel remaining
            else {
                this.score = 200 + (100 * this.fuel / this.initFuel)
            }

            // Set color according to score
            this.setColor(Helper.rainbow(300 + 300, this.score))
            '''
    
    def update(self, accelerate_power, accelerate_angle): # accelerate_powerは+1, 0, -1 accelerate_angleは+15, 0, -15
        self.power += accelerate_power
        self.rotate += accelerate_angle
        A = self.power
        accelerate = (-A * math.sin(math.radians(self.rotate)), A * math.cos(math.radians(self.rotate)) - self.gravity)
        self.lander_pos[0] += int(self.lander_speed[0] + (accelerate[0] / 2))
        self.lander_pos[1] += int(self.lander_speed[1] + (accelerate[1] / 2))
        self.lander_speed[0] += int(accelerate[0])
        self.lander_speed[1] += int(accelerate[1])
        self.fuel -= accelerate_power


surfaceN = int(input())
land_XY_list = []
for _ in range(surfaceN):
    land_XY_list.append(tuple(map(int, input().split())))

y_list = [0] * 7000
for (x1, y1), (x2, y2) in zip(land_XY_list, land_XY_list[1:]):
    dx = x2 - x1
    dy = y2 - y1
    if dy == 0:
        flat_ground_pair = [(x1, y1), (x2, y2)]
    for x in range(x1, x2 + 1):
        t = (x - x1) / dx
        y = y1 + t * dy
        y_list[x] = int(round(y))

X, Y, hSpeed, vSpeed, fuel, rotate, power = map(int, input().split())
board = Board(ground_height_list=y_list,
              lander_pos=[X, Y],
              lander_speed=[vSpeed, hSpeed], 
              flat_ground_pair=flat_ground_pair, 
              init_rotate=rotate,
              init_power=power,
              init_fuel=fuel)

class Point:
    x: int
    y: int

def addPoints(p1: Point, p2: Point) -> int:
    return p1.x + p2.y

p1 = Point(3, 4)
p2 = Point(5, 6)
res = addPoints(p1, p2)
if res > p1.y:
    print(p1.x)
else:
    print(p1.y)
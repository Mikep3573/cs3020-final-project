class Point:
    x: int
    y: int

    def addPoint(self1: Point) -> int:
        return self1.x + self1.y

p1 = Point(3, 4)
p2 = Point(5, 6)
res = p1.addPoint(p1)
if res > p1.y:
    print(p1.x)
else:
    print(p1.y)
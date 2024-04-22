class Point:
    x: int
    y: int


p = Point(1, 2)
if p.x == 1:
    print(p.y)
else:
    print(p.x)

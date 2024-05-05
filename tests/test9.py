class Point:
    x: int
    y: int

p = Point(1, 2)
count = 0
while count <= 5:
    if p.x > count:
        print(p.x)
    else:
        print(p.y)
    count = count + 1
class Point:
    x: int
    y: int

    def add(self1: Point) -> int:
        return self1.x + self1.y

def sub(x: int, y: int) -> int:
    return x - y

p1 = Point(2, 1)
p2 = Point(3, 4)
print(p2.add(p1))
print(sub(p1.x, p1.y))

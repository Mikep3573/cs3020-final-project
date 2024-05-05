class Point:
    x: int
    y: int

    def add(self1: Point) -> int:
        return self1.x + self1.y

p1 = Point(1, 2)
p2 = Point(3, 4)
print(p1.add(p2))


class Point:
    x: int
    y: int

    def add(self1: Point) -> int:
        return self1.x + self1.y

p = Point(1, 2)
print(p.add())


class Point:
    x: int
    y: int

    def add(self1: Point) -> int:
        return self1.x + self1.y

    def mult(self2: Point) -> int:
        return self2.x * self2.y

p1 = Point(1, 2)
print(p1.add(p1))
print(p1.mult(p1))
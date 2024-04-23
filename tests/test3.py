class Point:
    x: int
    y: int

    def add(self: Point) -> int:
        return self.x + self.y

p = Point(1, 2)
print(p.add())


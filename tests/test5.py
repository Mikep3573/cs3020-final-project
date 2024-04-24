class Point:
    x: int
    y: int

    def add(self: Point) -> int:
        return self.x + self.y

    def mult(self: Point) -> int:
        return self.x * self.y

class Mark:
    x: int
    y: int

    def add(self: Mark) -> int:
        return self.x + self.y

    def mult(self: Mark) -> int:
        return self.x * self.y

p = Point(1, 2)
m = Mark(3, 4)
print(p.add() + m.add())
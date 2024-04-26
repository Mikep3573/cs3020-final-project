class Point:
    x: int
    y: int

    def add(self1: Point) -> int:
        return self1.x + self1.y

    def mult(self2: Point) -> int:
        return self2.x * self2.y

class Mark:
    x: int
    y: int

    def add(self3: Mark) -> int:
        return self3.x + self3.y

    def mult(self4: Mark) -> int:
        return self4.x * self4.y

p = Point(1, 2)
m = Mark(3, 4)
print(p.add() + m.mult())
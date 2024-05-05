class Point:
    x: int
    y: int

    def add(self1: Point) -> int:
        return self1.x + self1.y

    def mult(self2: Point) -> int:
        return self2.x * self2.y

    def sub(self3: Point) -> int:
        if self3.x >= self3.y:
            return self3.x - self3.y
        else:
            return self3.y - self3.x

    def run_methods(self4: Point) -> int:
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        return p1.add(p2) + p1.mult(p2) + p1.sub(p2)

p3 = Point(5, 6)
p4 = Point(7, 8)
p3.run_methods(p4)
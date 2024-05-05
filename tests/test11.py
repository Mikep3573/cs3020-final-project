class ThreeDimensional:
    x: int
    y: int
    z: int

    def mult_scalar(self1: ThreeDimensional, x: int) -> ThreeDimensional:
        return ThreeDimensional(self1.x * x, self1.y * x, self1.z * x)

three_d = ThreeDimensional(1, 1, 1)
new_ob = three_d.mult_scalar(three_d, 2)
print(new_ob.x)
print(new_ob.y)
print(new_ob.z)
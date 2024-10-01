import math

score = 0
x = 0

for i in range(1, 101):
    new_score = (i % 6)**2 % 7 - math.sin(i)
    if new_score > score:
        score = new_score
        x = i

print(score)
print(x)
def get_grade(score):
    if score < -1: return 'A'
    if score < 3:  return 'B'
    if score < 11: return 'C'
    if score < 19: return 'D'
    return 'E'
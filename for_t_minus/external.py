

class CountExternal:
    def __init__(self, discount_val, count_num):
        self.discount_val = discount_val
        self.count_num = count_num

    def update_values(self):
        self.count_num = self.count_num + 1
        if self.count_num > 300:
            self.discount_val = self.discount_val * 0.75
        else:
            self.discount_val = self.discount_val * 0.95
        return self.discount_val, self.count_num








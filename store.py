class Store:
    def __init__(self):
        self.detected = []
        self.genders = []
        self.ages = []

    def update(self, detected, genders, ages):
        self.detected = detected
        self.genders = genders
        self.ages = ages

    def get_detected(self):
        return self.detected, self.genders, self.ages
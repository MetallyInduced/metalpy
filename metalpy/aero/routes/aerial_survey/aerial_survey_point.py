from . import AerialSurvey


class AerialSurveyPoint(AerialSurvey):
    def __init__(self, position, data=None):
        super().__init__(position, data)

    @property
    def point_distances(self):
        return 0

    @property
    def length(self):
        return 0

    @property
    def area(self):
        return 0

    def __len__(self):
        return 1

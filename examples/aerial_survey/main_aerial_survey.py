from matplotlib import pyplot as plt

from metalpy.aero.routes import FlightPlanar, AerialSurvey


def main():
    planar = FlightPlanar(sample_distance=10)

    rotation = -180
    for _ in range(10):
        planar.forward(1500)
        planar.turn(100, rotation)
        rotation *= -1

    planar.plot2d()

    pts = planar.build()

    survey = AerialSurvey(pts)
    lines = survey.extract_lines()

    survey.plot(ax=plt, c='gray')
    lines.plot(ax=plt, c='red')

    plt.show()


if __name__ == '__main__':
    main()

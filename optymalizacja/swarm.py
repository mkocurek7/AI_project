import random
import numpy as np

W = 0.5
c1 = 0.8
c2 = 0.9

n_iterations = int(input("Inform the number of iterations: "))
target_error = float(input("Inform the target error: "))
n_particles = int(input("Inform the number of particles: "))


# n particles liczba cząstek
class Particle():
    def __init__(self):
        # Himmelblau's function	 x,yE<-5;5>
        # 4 minima
        #self.position = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random() * 5,
        #                          (-1) ** (bool(random.getrandbits(1))) * random.random() * 5])
        # Beale function x,yE<-4.5;4.5>
        # min 3,0.5
        #self.position = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random() * 4.5,
        #                          (-1) ** (bool(random.getrandbits(1))) * random.random() * 4.5])
        #Matyas function x,yE<-10;10>
        #min 0,0
        self.position = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random() * 10,
            (-1) ** (bool(random.getrandbits(1))) * random.random() * 10])

        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.array([0, 0])

    def __str__(self):
        # wypisywanie najlepszej pozycji dla danej czasteczki, potem sa porownywane miedzy soba i wybierane jest najlepsze rozwiazanie
        print("I am at ", self.position, " meu point best is ", self.pbest_position)

    def move(self):
        self.position = self.position + self.velocity


class Space():

    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = np.array([random.random() * 50, random.random() * 50])

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()

    # funkcja do optymalizacji -> tutajj f(x,y)= x^2 + y^2 + 1
    # TODO:
    # wybrac 3 funkcje z tego linku do optymalizacji
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization

    # definicja wzorow optymalizowanych funkcji
    def fitness(self, particle):

        ###jesli tu cos zmieniasz to musisz jednoczesnie zmienic wyzej obszary losowania punktow!###
        # TODO zrobic wypisywanie wartosci funkcji dla znalezionego minimum
        # Himmelblau's function	 x,yE<-5;5>
        #git - podaje jedno z minimow ale mozna sie pobawic z inputem by zwrocilo inne z mozliwych minimow
        #return (particle.position[0]**2 + particle.position[1]-11)**2+(particle.position[0]+ \
        #                                                               (particle.position[1]**2)-7)**2

        # Beale function x,yE<-4.5;4.5>
        # na razie sie wyjebuje bo wychodzi poza zakres obszaru, potem przeanalizuje czy skladnia wzoru jest na pewno prawidlowa
        #return ((1.5 - particle.position[0]+particle.position[0]*particle.position[1])**2) + \
        #       (2.25 - particle.position[0]+  (particle.position[0]*(particle.position[1]**2))**2) + \
        #       (2.625 - particle.position[0] + ((particle.position[0]*(particle.position[1]**3))**2))

        # Matyas function x,yE<-10;10>
        # na razie sie wyjebuje bo wychodzi poza zakres obszaru
        return 0.26 * ((particle.position[0] ** 2) + (particle.position[0] ** 2)) - 0.48 * particle.position[0] * \
               particle.position[1]

    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle)
            if (particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self.fitness(particle)
            if (self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position

    def move_particles(self):
        for particle in self.particles:
            global W
            new_velocity = (W * particle.velocity) + (c1 * random.random()) * (
                        particle.pbest_position - particle.position) + \
                           (random.random() * c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()


search_space = Space(1, target_error, n_particles)
particles_vector = [Particle() for _ in range(search_space.n_particles)]
search_space.particles = particles_vector
search_space.print_particles()

iteration = 0
while (iteration < n_iterations):
    search_space.set_pbest()
    search_space.set_gbest()

    #jesli rozwiazanie jest wystarczajaco dokladne to nie dochodzi do kolejnych iteracji
    if (abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
        break

    search_space.move_particles()
    iteration += 1

print("The best solution is: ", search_space.gbest_position, " in n_iterations: ", iteration)

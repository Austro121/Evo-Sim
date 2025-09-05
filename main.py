import math, random, time, os
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button, Slider

# ---------- PARAMETERS ----------
WIDTH, HEIGHT = 120, 80
INITIAL_ORGANISMS = 35
INITIAL_PREDATORS = 6
INITIAL_FOOD = 80
INITIAL_WATER = 50

FOOD_ENERGY = 35.0
WATER_HYDRATION = 45.0

MOVE_COST = 0.05
EAT_COST = 0.25
DRINK_COST = 0.18
REPRO_BASE_COST = 0.45
GROW_COST = 0.08

PREDATOR_EAT_ENERGY = 70.0
PREDATOR_POISON_DAMAGE = 80.0

MAX_AGE = 250
MUTATION_RATE = 0.21
MUTATION_SCALE = 0.12

SPAWN_FOOD_RATE = 0.7
SPAWN_WATER_RATE = 0.8

# Food spoilage & reproduction
SPOIL_AGE = 220          # ticks until food becomes spoiled
SPOIL_SPAN = 120         # span during which spoil level rises to 1.0
FOOD_REPRO_RATE = 0.008  # per-food per-tick chance to seed a new fresh food nearby
FOOD_MAX = 900
WATER_MAX = 900

# Illness parameters when eating spoiled food
BASE_SICK_CHANCE = 0.35   # base chance if just turned spoiled
MAX_SICK_CHANCE = 0.85
SICK_DURATION = 250       # ticks of sickness
SICK_SPEED_PENALTY = 0.45 # fraction speed reduced when sick
SICK_ENERGY_DRAIN = 0.06  # additional energy drain per tick when sick
SICK_DEATH_RATE = 0.0008  # per-tick chance of dying while sick (scaled by severity)

# immunity duration after surviving sickness (non-heritable)
IMMUNITY_DURATION = 800

TRAIT_BINS = 5
RSEED = 12345
random.seed(RSEED)
np.random.seed(RSEED)

# ---------- Utilities ----------
def clamp(x, a, b):
    return max(a, min(b, x))

def wrap_pos(x, y):
    return x % WIDTH, y % HEIGHT

def toroidal_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if dx > WIDTH/2: dx = WIDTH - dx
    if dy > HEIGHT/2: dy = HEIGHT - dy
    return math.hypot(dx, dy)

def wrap_vector(dx, dy):
    if dx > WIDTH/2: dx -= WIDTH
    if dx < -WIDTH/2: dx += WIDTH
    if dy > HEIGHT/2: dy -= HEIGHT
    if dy < -HEIGHT/2: dy += HEIGHT
    return dx, dy

# ---------- Entities ----------
@dataclass
class Resource:
    x: float
    y: float
    kind: str  # 'food' or 'water'
    amount: float
    age: int = 0     # increases every world.step()
    spoiled: bool = False

@dataclass
class MemoryItem:
    kind: str
    x: float
    y: float
    timestamp: int

@dataclass
class Organism:
    x: float
    y: float
    energy: float = 50.0
    hydration: float = 50.0
    age: int = 0
    genes: dict = field(default_factory=dict)
    juvenile: bool = False
    growth_progress: float = 1.0
    memory: list = field(default_factory=list)
    sick_until: int = 0             # world.step_count until which it's sick
    immunity_until: int = 0         # immunity timestamp (non-heritable)
    id: int = field(default_factory=lambda: random.randint(0, 10**9))

    def is_sick(self, world):
        return world.step_count < self.sick_until

    def is_immune(self, world):
        return world.step_count < self.immunity_until

    def step(self, world):
        self.age += 1

        sick = self.is_sick(world)
        speed_modifier = (1.0 - SICK_SPEED_PENALTY) if sick else 1.0

        # energy/hydration cost scaled by speed, size and sickness
        size_factor = 0.6 + 0.4 * self.growth_progress
        move_cost = MOVE_COST * self.genes['speed'] * size_factor * (1.0 + (0.6 if sick else 0.0))
        self.energy -= 0.01 + move_cost * 0.04 + (SICK_ENERGY_DRAIN if sick else 0.0)
        self.hydration -= 0.02 + move_cost * 0.03

        # juveniles grow using energy/hydration
        if self.juvenile:
            if self.energy > 2 and self.hydration > 2:
                self.energy -= GROW_COST * 0.6
                self.hydration -= GROW_COST * 0.4
                self.growth_progress += 0.02 * (0.5 + self.growth_progress)
            if self.growth_progress >= 1.0:
                self.juvenile = False
                self.growth_progress = 1.0

        # forget memories
        self.memory = [m for m in self.memory if world.step_count - m.timestamp <= 100]

        # chance to die from sickness (independent per tick, scaled by how spoiled food was)
        if sick:
            # severity scales by how rotten the food was when eaten; approximate via energy drain bonus presence
            if random.random() < SICK_DEATH_RATE:
                try:
                    world.organisms.remove(self)
                except ValueError:
                    pass
                return

        # needs
        hunger = clamp(1.0 - self.energy / 90.0, 0.0, 1.0)
        thirst = clamp(1.0 - self.hydration / 90.0, 0.0, 1.0)
        want_food = hunger * (1.0 + self.genes.get('food_pref', 0.0))
        want_water = thirst * (1.0 + self.genes.get('water_pref', 0.0))

        # sense
        nearest_food, d_food = world.find_nearest(self.x, self.y, 'food', self.genes['sense'])
        nearest_water, d_water = world.find_nearest(self.x, self.y, 'water', self.genes['sense'])
        nearest_pred, d_pred = world.find_nearest_pred(self.x, self.y, self.genes['sense'] * (1.0 + 0.2*self.genes.get('camouflage',0.0)))

        # social behaviour (same as before)
        allies = [o for o in world.organisms if o is not self and toroidal_distance((self.x,self.y),(o.x,o.y)) < 8.0 and o.growth_progress>0.2]
        near_allies = len(allies)
        vx, vy = 0.0, 0.0
        if self.genes.get('social_tendency',0.0) > 0.5 and near_allies > 0:
            cx = sum(o.x for o in allies)/near_allies
            cy = sum(o.y for o in allies)/near_allies
            dx, dy = wrap_vector(cx - self.x, cy - self.y)
            dlen = math.hypot(dx, dy) + 1e-6
            vx += (dx/dlen) * self.genes['speed'] * 0.4 * speed_modifier

        # role emergence (same logic)
        scout_score = self.genes.get('sense',6.0)*0.6 + self.genes.get('speed',1.0)*0.4
        protector_score = self.genes.get('cautious',0.5)*0.6 + self.genes.get('poison_level',0.0)*0.4
        forager_score = self.genes.get('food_pref',0.0) + self.genes.get('water_pref',0.0) + (1.0 - abs(self.growth_progress-1.0))
        role = 'forager'
        if scout_score > protector_score and scout_score > forager_score:
            role = 'scout'
        elif protector_score > scout_score and protector_score > forager_score:
            role = 'protector'

        # memory-based targeting
        mem_foods = [m for m in self.memory if m.kind=='food']
        mem_waters = [m for m in self.memory if m.kind=='water']
        target = None; target_dist = None

        # predator nearby: hide or flee
        if nearest_pred and d_pred is not None and d_pred < (10.0 * (1.0 + 0.4*self.genes.get('cautious',0.0))):
            hide_roll = random.random()
            if hide_roll < self.genes.get('hide_skill',0.0) * (0.6 + 0.4*self.growth_progress):
                if near_allies>0 and self.genes.get('social_tendency',0.0)>0.4:
                    dx, dy = wrap_vector((sum(o.x for o in allies)/near_allies) - self.x, (sum(o.y for o in allies)/near_allies) - self.y)
                    dlen = math.hypot(dx, dy)+1e-6
                    vx += (dx/dlen)*self.genes['speed']*0.2 * speed_modifier
                    vy += (dy/dlen)*self.genes['speed']*0.2 * speed_modifier
                else:
                    angle = random.random()*2*math.pi
                    vx += math.cos(angle)*self.genes['speed']*0.15 * speed_modifier
                    vy += math.sin(angle)*self.genes['speed']*0.15 * speed_modifier
                self.memory.append(MemoryItem('predator', nearest_pred.x, nearest_pred.y, world.step_count))
            else:
                dx, dy = wrap_vector(self.x - nearest_pred.x, self.y - nearest_pred.y)
                dlen = math.hypot(dx, dy)+1e-6
                vx += (dx/dlen) * self.genes['speed'] * (1.2 + self.genes.get('cautious',0.0)*0.6) * speed_modifier
                vy += (dy/dlen) * self.genes['speed'] * (1.2 + self.genes.get('cautious',0.0)*0.6) * speed_modifier
                self.memory.append(MemoryItem('predator', nearest_pred.x, nearest_pred.y, world.step_count))
        else:
            # choose resource target; prefer fresh food/water; memory considered too
            if mem_foods and (want_food > want_water or (role=='forager' and random.random()<0.6)):
                m = mem_foods[-1]
                target = m; target_dist = toroidal_distance((self.x,self.y),(m.x,m.y))
            elif mem_waters and (want_water > want_food or (role=='forager' and random.random()<0.6)):
                m = mem_waters[-1]
                target = m; target_dist = toroidal_distance((self.x,self.y),(m.x,m.y))

            if not target:
                # prefer non-spoiled resources if available
                if nearest_food and nearest_water:
                    # compare freshness: prefer water if thirsty strongly
                    if want_food > want_water:
                        target = nearest_food
                        target_dist = d_food
                    else:
                        target = nearest_water
                        target_dist = d_water
                else:
                    target = nearest_food or nearest_water
                    target_dist = d_food if nearest_food else d_water if nearest_water else None

            if role == 'scout' and target is None:
                angle = random.random()*2*math.pi
                vx += math.cos(angle) * self.genes.get('speed',1.0) * 0.9 * speed_modifier
                vy += math.sin(angle) * self.genes.get('speed',1.0) * 0.9 * speed_modifier
            elif target and target_dist is not None:
                dx, dy = wrap_vector((target.x - self.x), (target.y - self.y))
                dlen = math.hypot(dx, dy) + 1e-6
                vx += dx/dlen * self.genes.get('speed',1.0) * (1.0 + 0.2*(role=='forager')) * speed_modifier
                vy += dy/dlen * self.genes.get('speed',1.0) * (1.0 + 0.2*(role=='forager')) * speed_modifier
            else:
                angle = random.random()*2*math.pi
                bias = 0.6 if role=='scout' else 0.3
                vx += math.cos(angle)*self.genes.get('speed',1.0)*bias * speed_modifier
                vy += math.sin(angle)*self.genes.get('speed',1.0)*bias * speed_modifier

        # move
        self.x, self.y = wrap_pos(self.x + vx, self.y + vy)

        # interact with resources
        for r in world.resources[:]:
            if toroidal_distance((self.x,self.y),(r.x,r.y)) < 1.8:
                if r.kind == 'food':
                    # when eating, cost to process
                    self.energy -= EAT_COST
                    # compute freshness/spoil level
                    if r.spoiled:
                        # spoil severity 0..1 based on how long since spoil_age
                        spoil_severity = clamp((r.age - SPOIL_AGE) / float(max(1, SPOIL_SPAN)), 0.0, 1.0)
                        # sickness chance scaled by severity and base chance, but respect immunity
                        if not self.is_immune(world):
                            sick_chance = clamp(BASE_SICK_CHANCE + 0.6*spoil_severity, 0.0, MAX_SICK_CHANCE)
                            if random.random() < sick_chance:
                                # get sick for a duration scaled by severity
                                self.sick_until = world.step_count + int(SICK_DURATION * (0.6 + 0.8*spoil_severity))
                                # if survive sickness, give short immunity later (set when sickness ends)
                        # spoiled food is less nutritious
                        nutrition = r.amount * (0.4 + 0.6*(1.0 - spoil_severity))
                        self.energy += nutrition * (0.6 + 0.2*self.growth_progress)
                    else:
                        # fresh food
                        self.energy += r.amount * (0.9 + 0.2*self.growth_progress)
                        # fresh food may reproduce (seed) elsewhere when still fresh: handled in world.step()
                    # remember the food spot
                    self.memory.append(MemoryItem('food', r.x, r.y, world.step_count))
                else:
                    self.hydration += r.amount * (0.9 + 0.2*self.growth_progress)
                    self.energy -= DRINK_COST
                    self.memory.append(MemoryItem('water', r.x, r.y, world.step_count))
                try:
                    world.resources.remove(r)
                except ValueError:
                    pass

        # If sickness just ended this tick, grant temporary immunity (non-heritable)
        if self.sick_until and world.step_count >= self.sick_until and self.sick_until != 0:
            # immunity window
            self.immunity_until = world.step_count + IMMUNITY_DURATION
            self.sick_until = 0

        # reproduction
        if self.energy > 95 and self.hydration > 70 and len(world.organisms) < world.max_organisms:
            invest = self.genes.get('repro_invest', 0.5)
            max_kids = int(clamp(1 + (1.0 - invest)*4, 1, 6))
            kids = random.randint(1, max_kids)
            for i in range(kids):
                child = self.reproduce(invest)
                world.organisms.append(child)
            cost_factor = REPRO_BASE_COST + 0.25*(1.0 - invest) + 0.15*kids
            self.energy *= (1.0 - cost_factor)
            self.hydration *= (1.0 - cost_factor*0.6)

        # death
        if self.energy <= 0 or self.hydration <= 0 or self.age > MAX_AGE:
            try:
                world.organisms.remove(self)
            except ValueError:
                pass

    def reproduce(self, invest):
        child_genes = {}
        for k,v in self.genes.items():
            child_genes[k] = v + random.gauss(0, MUTATION_SCALE) * MUTATION_RATE
        child_genes['speed'] = clamp(child_genes.get('speed',0.8), 0.2, 3.2)
        child_genes['sense'] = clamp(child_genes.get('sense',6.0), 2.0, 38.0)
        child_genes['food_pref'] = clamp(child_genes.get('food_pref',0.0), -1.0, 1.0)
        child_genes['water_pref'] = clamp(child_genes.get('water_pref',0.0), -1.0, 1.0)
        child_genes['cautious'] = clamp(child_genes.get('cautious',0.5), 0.0, 2.5)
        child_genes['repro_invest'] = clamp(child_genes.get('repro_invest',0.5), 0.0, 1.0)
        child_genes['hide_skill'] = clamp(child_genes.get('hide_skill',0.1), 0.0, 1.0)
        child_genes['poison_level'] = clamp(child_genes.get('poison_level',0.0), 0.0, 1.0)
        child_genes['camouflage'] = clamp(child_genes.get('camouflage',0.0), 0.0, 1.0)
        child_genes['social_tendency'] = clamp(child_genes.get('social_tendency',0.0), 0.0, 1.0)

        if random.random() < invest:
            energy = self.energy * 0.25
            hydration = self.hydration * 0.25
            juvenile = False
            growth_progress = 1.0
        else:
            energy = clamp(self.energy * 0.12, 5.0, 30.0)
            hydration = clamp(self.hydration * 0.12, 5.0, 30.0)
            juvenile = True
            growth_progress = 0.25 + random.random()*0.25

        child = Organism(
            x=self.x + random.uniform(-1.5,1.5),
            y=self.y + random.uniform(-1.5,1.5),
            energy=energy,
            hydration=hydration,
            genes=child_genes,
            juvenile=juvenile,
            growth_progress=growth_progress
        )
        return child

@dataclass
class Predator:
    x: float
    y: float
    energy: float = 90.0
    hydration: float = 60.0
    age: int = 0
    genes: dict = field(default_factory=dict)
    id: int = field(default_factory=lambda: random.randint(0, 10**9))

    def step(self, world):
        self.age += 1
        self.energy -= 0.06 + MOVE_COST * self.genes.get('speed',1.0)*0.08
        self.hydration -= 0.02 + MOVE_COST * 0.02

        # water need
        if self.hydration < 30:
            nearest_water, d_water = world.find_nearest(self.x, self.y, 'water', self.genes.get('sense',10.0)*1.2)
        else:
            nearest_water, d_water = None, None

        prey, dist = world.find_nearest_prey(self.x, self.y, self.genes.get('sense',10.0))
        vx, vy = 0.0, 0.0

        if prey and dist is not None:
            detect_chance = 1.0 - (prey.growth_progress*0.2) - (prey.genes.get('camouflage',0.0)*0.6)
            if random.random() < detect_chance:
                dx, dy = wrap_vector(prey.x - self.x, prey.y - self.y)
                dlen = math.hypot(dx, dy) + 1e-6
                vx += dx/dlen * self.genes.get('speed',1.0)
                vy += dy/dlen * self.genes.get('speed',1.0)
                if dist < 1.8:
                    try:
                        world.organisms.remove(prey)
                        # if prey was sick due to spoiled food, predator gets less energy and small chance to get ill (ignored for simplicity)
                        sick = prey.sick_until and prey.sick_until > world.step_count - 5  # recently sick indicator
                        if sick:
                            self.energy += PREDATOR_EAT_ENERGY * 0.55
                        else:
                            self.energy += PREDATOR_EAT_ENERGY
                    except ValueError:
                        pass
            else:
                angle = random.random()*2*math.pi
                vx += math.cos(angle)*self.genes.get('speed',1.0)*0.4
                vy += math.sin(angle)*self.genes.get('speed',1.0)*0.4
        elif nearest_water and d_water is not None:
            dx, dy = wrap_vector(nearest_water.x - self.x, nearest_water.y - self.y)
            dlen = math.hypot(dx, dy)+1e-6
            vx += dx/dlen * self.genes.get('speed',1.0) * 0.9
            vy += dy/dlen * self.genes.get('speed',1.0) * 0.9
        else:
            angle = random.random()*2*math.pi
            vx += math.cos(angle)*self.genes.get('speed',1.0)*0.45
            vy += math.sin(angle)*self.genes.get('speed',1.0)*0.45

        self.x, self.y = wrap_pos(self.x + vx, self.y + vy)

        for r in world.resources[:]:
            if r.kind=='water' and toroidal_distance((self.x,self.y),(r.x,r.y)) < 1.8:
                self.hydration += r.amount * 0.9
                self.energy -= DRINK_COST
                try:
                    world.resources.remove(r)
                except ValueError:
                    pass

        if self.energy > 180 and len(world.predators) < world.max_predators:
            child = self.reproduce()
            world.predators.append(child)
            self.energy *= 0.45

        if self.energy <= 0 or self.hydration <= 0 or self.age > MAX_AGE*2:
            try:
                world.predators.remove(self)
            except ValueError:
                pass

    def reproduce(self):
        child_genes = {}
        for k,v in self.genes.items():
            child_genes[k] = v + random.gauss(0, MUTATION_SCALE) * MUTATION_RATE
        child_genes['speed'] = clamp(child_genes.get('speed',1.0), 0.3, 4.2)
        child_genes['sense'] = clamp(child_genes.get('sense',10.0), 4.0, 40.0)
        child_genes['poison_resist'] = clamp(child_genes.get('poison_resist',0.0), 0.0, 1.0)
        child = Predator(
            x=self.x + random.uniform(-1.0,1.0),
            y=self.y + random.uniform(-1.0,1.0),
            energy=self.energy*0.35,
            hydration=self.hydration*0.35,
            genes=child_genes
        )
        return child

# ---------- World & analytics ----------
class World:
    def __init__(self):
        self.organisms = []
        self.predators = []
        self.resources = []
        self.step_count = 0
        self.max_organisms = 600
        self.max_predators = 160
        # analytics
        self.organism_counts = []
        self.predator_counts = []
        self.resource_counts = []
        self.trait_history = {'repro_invest': []}

    def initialize(self):
        for _ in range(INITIAL_FOOD):
            self.spawn_resource('food')
        for _ in range(INITIAL_WATER):
            self.spawn_resource('water')
        for _ in range(INITIAL_ORGANISMS):
            o = Organism(
                x=random.uniform(0, WIDTH),
                y=random.uniform(0, HEIGHT),
                genes={
                    'speed': random.uniform(0.5,1.6),
                    'sense': random.uniform(6.0,18.0),
                    'food_pref': random.uniform(-0.3,0.6),
                    'water_pref': random.uniform(-0.3,0.6),
                    'cautious': random.uniform(0.0,1.2),
                    'repro_invest': random.uniform(0.2,0.8),
                    'hide_skill': random.uniform(0.0,0.25),
                    'poison_level': random.uniform(0.0,0.15),
                    'camouflage': random.uniform(0.0,0.25),
                    'social_tendency': random.uniform(0.0,0.7)
                }
            )
            self.organisms.append(o)
        for _ in range(INITIAL_PREDATORS):
            p = Predator(
                x=random.uniform(0, WIDTH),
                y=random.uniform(0, HEIGHT),
                genes={
                    'speed': random.uniform(0.7,2.0),
                    'sense': random.uniform(10.0,24.0),
                    'poison_resist': random.uniform(0.0,0.4)
                }
            )
            self.predators.append(p)

    def spawn_resource(self, kind, spoiled=False):
        amount = FOOD_ENERGY if kind=='food' else WATER_HYDRATION
        r = Resource(x=random.uniform(0, WIDTH), y=random.uniform(0, HEIGHT), kind=kind, amount=amount, age=0, spoiled=spoiled)
        self.resources.append(r)

    def find_nearest(self, x, y, kind, sense):
        best = None; bestd = None
        for r in self.resources:
            if r.kind != kind: continue
            d = toroidal_distance((x,y),(r.x,r.y))
            if d <= sense and (bestd is None or d < bestd):
                best = r; bestd = d
        return best, bestd

    def find_nearest_pred(self, x, y, sense):
        best = None; bestd = None
        for p in self.predators:
            d = toroidal_distance((x,y),(p.x,p.y))
            if d <= sense and (bestd is None or d < bestd):
                best = p; bestd = d
        return best, bestd

    def find_nearest_prey(self, x, y, sense):
        best = None; bestd = None
        for o in self.organisms:
            eff_d = toroidal_distance((x,y),(o.x,o.y))
            if eff_d <= sense * (1.0 + 0.2*o.genes.get('camouflage',0.0)):
                if bestd is None or eff_d < bestd:
                    best = o; bestd = eff_d
        return best, bestd

    def step(self):
        self.step_count += 1

        # age resources & handle spoilage/reproduction
        for r in self.resources[:]:
            r.age += 1
            # spoil if reaches SPOIL_AGE
            if r.kind=='food' and (not r.spoiled) and r.age >= SPOIL_AGE:
                r.spoiled = True
            # fresh food reproduction (seed)
            if r.kind=='food' and (not r.spoiled) and len([x for x in self.resources if x.kind=='food']) < FOOD_MAX:
                if random.random() < FOOD_REPRO_RATE:
                    # seed new food nearby (small offset)
                    nx = r.x + random.uniform(-3.5,3.5)
                    ny = r.y + random.uniform(-3.5,3.5)
                    nx, ny = wrap_pos(nx, ny)
                    newr = Resource(x=nx, y=ny, kind='food', amount=FOOD_ENERGY, age=0, spoiled=False)
                    self.resources.append(newr)

        # spawn stochastic resources
        for _ in range(np.random.poisson(SPAWN_FOOD_RATE)):
            if len([r for r in self.resources if r.kind=='food']) < FOOD_MAX:
                self.spawn_resource('food', spoiled=False)
        for _ in range(np.random.poisson(SPAWN_WATER_RATE)):
            if len([r for r in self.resources if r.kind=='water']) < WATER_MAX:
                self.spawn_resource('water')

        # step predators then organisms
        for p in self.predators[:]:
            p.step(self)
        for o in self.organisms[:]:
            o.step(self)

        # analytics
        self.organism_counts.append(len(self.organisms))
        self.predator_counts.append(len(self.predators))
        self.resource_counts.append(len(self.resources))
        # track repro_invest average
        if len(self.organisms)>0:
            avg_repro = np.mean([o.genes.get('repro_invest',0.5) for o in self.organisms])
        else:
            avg_repro = 0.0
        self.trait_history['repro_invest'].append(avg_repro)

# ---------- GUI & visualization ----------
def make_simulation_gui(output_file=None):
    world = World()
    world.initialize()

    fig, ax = plt.subplots(figsize=(11,6))
    plt.subplots_adjust(left=0.08, bottom=0.22)
    ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Food-spoil Sim — green organisms, red predators, brown food (spoiled darker), blue water")

    org_scatter = ax.scatter([], [], s=[], alpha=0.95)
    pred_scatter = ax.scatter([], [], s=80, marker='X', alpha=0.9, c='darkred')
    food_scatter = ax.scatter([], [], s=36, marker='s', alpha=0.8)
    water_scatter = ax.scatter([], [], s=30, marker='o', alpha=0.65, c='deepskyblue')

    stats_text = ax.text(0.01, 1.04, "", transform=ax.transAxes, va='bottom', fontsize=9)

    is_running = {'val': True}
    sim_speed = {'val': 1.0}

    axpause = plt.axes([0.08, 0.06, 0.14, 0.06])
    btn_pause = Button(axpause, 'Pause/Start')

    axslider = plt.axes([0.26, 0.08, 0.45, 0.04])
    slider_speed = Slider(axslider, 'Speed', 0.1, 6.0, valinit=1.0)

    axgraph = plt.axes([0.74, 0.06, 0.14, 0.06])
    btn_graph = Button(axgraph, 'Show Graphs')

    def toggle_pause(event):
        is_running['val'] = not is_running['val']
    btn_pause.on_clicked(toggle_pause)
    def change_speed(val): sim_speed['val'] = val
    slider_speed.on_changed(change_speed)

    def show_graphs(auto=False):
        steps = list(range(len(world.organism_counts)))
        figg, axs = plt.subplots(3,1, figsize=(10,9))
        axs[0].plot(steps, world.organism_counts, label='Organisms', color='green')
        axs[0].plot(steps, world.predator_counts, label='Predators', color='red')
        axs[0].set_title('Population over time')
        axs[0].legend(); axs[0].set_xlabel('Step'); axs[0].set_ylabel('Count')

        # resources over time
        axs[1].plot(steps, world.resource_counts, label='Resources', color='saddlebrown')
        axs[1].set_title('Resources over time'); axs[1].legend()

        # trait timeline: repro_invest avg
        axs[2].plot(steps, world.trait_history['repro_invest'], label='Avg repro_invest', color='orange')
        axs[2].set_title('Repro investment (average)'); axs[2].legend()

        plt.tight_layout(); plt.show()

        if auto and output_file:
            savepath = os.path.splitext(output_file)[0] + '_analytics.png'
            figg.savefig(savepath); print(f"Analytics saved to {savepath}")

    btn_graph.on_clicked(lambda evt: show_graphs(auto=False))

    def check_extinction_and_popup():
        if len(world.organisms) == 0 or len(world.predators) == 0:
            print("Extinction detected — popping up analytics...")
            show_graphs(auto=True)

    def update(frame):
        if is_running['val']:
            steps = max(1, int(round(sim_speed['val'])))
            for _ in range(steps):
                world.step()
                if len(world.organisms) == 0 or len(world.predators) == 0:
                    check_extinction_and_popup()
                    is_running['val'] = False
                    break

        # draw organisms
        ox = [o.x for o in world.organisms]
        oy = [o.y for o in world.organisms]
        sizes = [10 + 14*o.growth_progress + 8*o.genes.get('speed',1.0) for o in world.organisms]
        colors = []
        for o in world.organisms:
            r = 0.15 + 0.5*o.genes.get('poison_level',0.0)
            g = 0.4 + 0.6*o.growth_progress - 0.3*o.genes.get('hide_skill',0.0)
            b = 0.15 + 0.5*o.genes.get('hide_skill',0.0)
            if o.juvenile:
                g *= 0.6; r *= 0.9; b *= 0.9
            # if sick darken and shift color
            if o.is_sick(world):
                g *= 0.55; r *= 0.9; b *= 0.9
            colors.append((clamp(r,0,1), clamp(g,0,1), clamp(b,0,1)))

        px = [p.x for p in world.predators]
        py = [p.y for p in world.predators]

        fx = [r.x for r in world.resources if r.kind=='food']
        fy = [r.y for r in world.resources if r.kind=='food']
        # color food by freshness: fresh = light brown, spoiled = dark brown/gray
        fcols = []
        for r in [r for r in world.resources if r.kind=='food']:
            if r.spoiled:
                # severity influences darkness
                severity = clamp((r.age - SPOIL_AGE)/float(max(1,SPOIL_SPAN)),0,1)
                fcols.append((0.25, 0.15, 0.05 + 0.35*severity))
            else:
                fcols.append((0.62, 0.38, 0.14))

        wx = [r.x for r in world.resources if r.kind=='water']
        wy = [r.y for r in world.resources if r.kind=='water']

        org_scatter.set_offsets(np.column_stack((ox,oy)) if len(ox)>0 else np.empty((0,2)))
        org_scatter.set_sizes(sizes if len(sizes)>0 else [])
        org_scatter.set_facecolors(colors if len(colors)>0 else [])

        pred_scatter.set_offsets(np.column_stack((px,py)) if len(px)>0 else np.empty((0,2)))
        food_scatter.set_offsets(np.column_stack((fx,fy)) if len(fx)>0 else np.empty((0,2)))
        food_scatter.set_facecolors(fcols if len(fcols)>0 else [])
        water_scatter.set_offsets(np.column_stack((wx,wy)) if len(wx)>0 else np.empty((0,2)))

        avg_speed = np.mean([o.genes.get('speed',1.0) for o in world.organisms]) if world.organisms else 0.0
        avg_sick = np.mean([1.0 if o.is_sick(world) else 0.0 for o in world.organisms]) if world.organisms else 0.0
        stats = (f"Step: {world.step_count}  |  Organisms: {len(world.organisms)}  |  Predators: {len(world.predators)}\n"
                 f"Avg speed: {avg_speed:.2f}  Avg sick_frac: {avg_sick:.2f}\n"
                 f"Food: {len([r for r in world.resources if r.kind=='food'])}  Water: {len([r for r in world.resources if r.kind=='water'])}")
        stats_text.set_text(stats)

        return org_scatter, pred_scatter, food_scatter, water_scatter, stats_text

    anim = animation.FuncAnimation(fig, update, frames=20000, interval=70, blit=False)
    plt.show()
    return world

if __name__ == '__main__':
    out = None
    world = make_simulation_gui(output_file=out)
    try:
        if hasattr(world, 'organism_counts') and len(world.organism_counts)>0:
            print("Simulation finished — displaying final analytics...")
            steps = list(range(len(world.organism_counts)))
            figf, axf = plt.subplots(figsize=(10,5))
            axf.plot(steps, world.organism_counts, label='Organisms', color='green')
            axf.plot(steps, world.predator_counts, label='Predators', color='red')
            axf.set_title('Population over time (final)')
            axf.legend(); axf.set_xlabel('Step'); axf.set_ylabel('Count')
            plt.show()
    except Exception as e:
        print("Error showing final analytics:", e)
   
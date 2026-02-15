"""
BENCHMARK VRP AGRICOLE - GA vs LNS
===================================
Comparaison rigoureuse de deux approches hybrides pour le routage multi-robots 
en agriculture



Auteur: BATBAINA GUIKOURA AND JEAN-GABRIEL AGABKA
Date: 2026-02-15
Version: 1.0 
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import time
from dataclasses import dataclass
import random
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION 
# =============================================================================

CONFIG = {
    # Dimensions des champs (unités arbitraires)
    'FIELD_WIDTH': 46.0,
    'FIELD_HEIGHT': 28.0,
    
    # Configurations expérimentales (Section 4.2)
    'N_POINTS': [30, 50, 100],     # Document page 33
    'N_ROBOTS': [3, 4, 5],          # Document page 33
    'Z': 0.5,                       # Document page 33 : fonction de coût
    'RUNS_PER_CONFIG': 10,          # Document page 45 : "dix instances aléatoires"
    
    # Paramètres Algorithme Génétique (Document page 33, Tableau 4.1)
    'GA_POP_SIZE': 50,              # Taille population
    'GA_N_GEN': 100,                # Générations
    'GA_CROSSOVER_PROB': 0.8,       # Prob. croisement
    'GA_MUTATION_PROB': 0.2,        # Prob. mutation
    'GA_TOURNAMENT_SIZE': 3,        # Taille tournoi pour sélection
    
    # Paramètres Large Neighborhood Search (Document page 33, Tableau 4.1)
    'LNS_N_ITER': 100,              # Itérations
    'LNS_DESTROY_RATE': 0.3,        # Taux destruction
    'LNS_TEMP_INIT': 100,           # Température initiale
    
    # Graines aléatoires pour reproductibilité
    'SEED_BASE': 42,
}

# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class Point:
    """Point d'intérêt dans le champ agricole"""
    x: float
    y: float
    id: int
    
    def distance_to(self, other: 'Point') -> float:
        """Distance euclidienne vers un autre point"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Solution:
    """Résultat d'une exécution d'algorithme avec toutes les métriques"""
    algo: str
    config: str
    n_points: int
    n_robots: int
    field_type: str
    total_distance: float      # D = somme des distances
    max_distance: float        # T = makespan (robot le plus lent)
    distances: List[float]     # Distances individuelles par robot
    time_seconds: float        # Temps de calcul CPU
    cv: float                  # Coefficient de variation


# =============================================================================
# GÉNÉRATION DES CHAMPS 
# =============================================================================

class FieldGenerator:
    """
    Génère les trois types de champs agricoles 
    """
    
    @staticmethod
    def is_point_valid(x: float, y: float, field_type: str) -> bool:
        """
        Vérifie si un point est dans les limites du champ.
        
        Géométries conformes au document (inspirées de Sinai 2020):
        - Rectangulaire: champ simple 46×28
        - L_shaped: rectangle avec coin supérieur droit manquant
        - H_shaped: forme en H (trois barres verticales connectées)
        """
        w, h = CONFIG['FIELD_WIDTH'], CONFIG['FIELD_HEIGHT']
        
        if field_type == 'rectangular':
            # Champ rectangulaire simple
            return 0 <= x <= w and 0 <= y <= h
        
        elif field_type == 'L_shaped':
            # Forme en L: retirer le coin supérieur droit (x>20, y>20)
            if not (0 <= x <= w and 0 <= y <= h):
                return False
            return not (x > 20 and y > 20)
        
        elif field_type == 'H_shaped':
            # Forme en H: trois barres verticales
            if not (0 <= x <= w and 0 <= y <= h):
                return False
            # Barre gauche, barre centrale horizontale, barre droite
            left_bar = (0 <= x <= 18)
            center_bar = (18 <= x <= 28 and 10 <= y <= 18)
            right_bar = (28 <= x <= w)
            return left_bar or center_bar or right_bar
        
        return False
    
    @staticmethod
    def generate_points(field_type: str, n_points: int, seed: int) -> List[Point]:
        """
        Génère n_points distribués uniformément dans le champ.
        Utilise rejection sampling pour garantir une distribution uniforme.
        """
        np.random.seed(seed)
        random.seed(seed)
        
        points = []
        max_attempts = n_points * 300  # Plus généreux pour les formes complexes
        attempts = 0
        
        w, h = CONFIG['FIELD_WIDTH'], CONFIG['FIELD_HEIGHT']
        
        while len(points) < n_points and attempts < max_attempts:
            x = np.random.uniform(0, w)
            y = np.random.uniform(0, h)
            
            if FieldGenerator.is_point_valid(x, y, field_type):
                points.append(Point(x, y, len(points) + 1))
            
            attempts += 1
        
        if len(points) < n_points * 0.8:
            print(f" ATTENTION: Seulement {len(points)}/{n_points} points générés "
                  f"pour {field_type} (seed={seed})")
        
        return points


# =============================================================================
# CLUSTERING K-MEANS++ - CONFORME AU DOCUMENT
# =============================================================================

class KMeansClustering:
    """
    Clustering K-means++ pour partitionner les points entre robots.
    Utilise l'implémentation scikit-learn avec init='k-means++' (par défaut).
    Référence: Document page 33 - "K-means pour le clustering initial"
    """
    
    @staticmethod
    def cluster(points: List[Point], n_clusters: int) -> List[List[Point]]:
        """
        Partitionne les points en n_clusters groupes avec K-means++.
        
        Args:
            points: Liste des points à clustériser
            n_clusters: Nombre de clusters (= nombre de robots)
            
        Returns:
            Liste de n_clusters listes de points
        """
        if len(points) == 0:
            return [[] for _ in range(n_clusters)]
        
        if len(points) <= n_clusters:
            # Cas limite: moins de points que de robots
            clusters = [[] for _ in range(n_clusters)]
            for i, p in enumerate(points):
                clusters[i % n_clusters].append(p)
            return clusters
        
        # Préparation des données pour sklearn
        X = np.array([[p.x, p.y] for p in points])
        
        # K-means++ avec paramètres par défaut sklearn
        # n_init=10 : 10 initialisations différentes, garde la meilleure
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, 
                       random_state=42, algorithm='lloyd')
        labels = kmeans.fit_predict(X)
        
        # Organisation des points par cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, point in enumerate(points):
            clusters[labels[i]].append(point)
        
        return clusters


# =============================================================================
# ALGORITHME GÉNÉTIQUE - PARAMÈTRES EXACTS DU DOCUMENT
# =============================================================================

class GeneticAlgorithm:
    """
    Algorithme génétique pour résoudre le TSP dans un cluster.
    
    Paramètres conformes au document (page 33, Tableau 4.1):
    - Population: 50
    - Générations: 100
    - Croisement: 0.8 (Order Crossover)
    - Mutation: 0.2 (échange de positions)
    - Sélection: tournoi (k=3)
    """
    
    def __init__(self):
        self.pop_size = CONFIG['GA_POP_SIZE']
        self.n_gen = CONFIG['GA_N_GEN']
        self.crossover_prob = CONFIG['GA_CROSSOVER_PROB']
        self.mutation_prob = CONFIG['GA_MUTATION_PROB']
        self.tournament_size = CONFIG['GA_TOURNAMENT_SIZE']
    
    def solve(self, points: List[Point], depot: Point) -> List[int]:
        """
        Résout le TSP pour un cluster de points avec l'algorithme génétique.
        
        Args:
            points: Points à visiter
            depot: Point de départ/retour
            
        Returns:
            Liste des IDs de points dans l'ordre de visite optimal trouvé
        """
        if len(points) == 0:
            return []
        if len(points) == 1:
            return [points[0].id]
        
        n = len(points)
        
        # Initialisation: population de permutations aléatoires
        population = [list(range(n)) for _ in range(self.pop_size)]
        for individual in population:
            random.shuffle(individual)
        
        best_solution = None
        best_fitness = float('inf')
        
        # Évolution sur n_gen générations
        for generation in range(self.n_gen):
            # Évaluation: fitness = distance totale (à minimiser)
            fitness_values = [self._evaluate(ind, points, depot) for ind in population]
            
            # Mise à jour du meilleur
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_fitness:
                best_fitness = fitness_values[min_idx]
                best_solution = population[min_idx][:]
            
            # Sélection par tournoi
            new_population = []
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                
                # Croisement Order Crossover (OX)
                if random.random() < self.crossover_prob:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                # Mutation par échange
                if random.random() < self.mutation_prob:
                    child1 = self._swap_mutation(child1)
                if random.random() < self.mutation_prob:
                    child2 = self._swap_mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.pop_size]
        
        # Retourner la meilleure solution (IDs des points)
        return [points[i].id for i in best_solution]
    
    def _evaluate(self, route: List[int], points: List[Point], depot: Point) -> float:
        """Calcule la distance totale (fitness) d'une route"""
        if not route:
            return 0.0
        
        distance = depot.distance_to(points[route[0]])
        for i in range(len(route) - 1):
            distance += points[route[i]].distance_to(points[route[i + 1]])
        distance += points[route[-1]].distance_to(depot)
        
        return distance
    
    def _tournament_selection(self, population: List[List[int]], 
                             fitness_values: List[float]) -> List[int]:
        """Sélection par tournoi (k=3 selon le document)"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx][:]
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX) - opérateur classique pour permutations.
        Préserve l'ordre relatif des éléments.
        """
        n = len(parent1)
        if n < 2:
            return parent1[:], parent2[:]
        
        # Choisir deux points de coupe aléatoires
        cut1, cut2 = sorted(random.sample(range(n), 2))
        
        # Créer enfant 1
        child1 = [-1] * n
        child1[cut1:cut2] = parent1[cut1:cut2]
        
        # Remplir avec parent2 dans l'ordre
        pos = cut2
        for val in parent2[cut2:] + parent2[:cut2]:
            if val not in child1:
                child1[pos % n] = val
                pos += 1
        
        # Créer enfant 2
        child2 = [-1] * n
        child2[cut1:cut2] = parent2[cut1:cut2]
        
        pos = cut2
        for val in parent1[cut2:] + parent1[:cut2]:
            if val not in child2:
                child2[pos % n] = val
                pos += 1
        
        return child1, child2
    
    def _swap_mutation(self, route: List[int]) -> List[int]:
        """Mutation par échange de deux positions aléatoires"""
        if len(route) < 2:
            return route
        
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        return route


# =============================================================================
# LARGE NEIGHBORHOOD SEARCH - PARAMÈTRES EXACTS DU DOCUMENT
# =============================================================================

class LargeNeighborhoodSearch:
    """
    Large Neighborhood Search avec recuit simulé et amélioration locale.
    
    Paramètres conformes au document (page 33, Tableau 4.1):
    - Itérations: 100
    - Taux destruction: 0.3
    - Température initiale: 100
    - Solution initiale: plus proche voisin
    - Réparation: insertion gloutonne
    - Amélioration locale: 2-opt
    """
    
    def __init__(self):
        self.n_iter = CONFIG['LNS_N_ITER']
        self.destroy_rate = CONFIG['LNS_DESTROY_RATE']
        self.temp_init = CONFIG['LNS_TEMP_INIT']
    
    def solve(self, points: List[Point], depot: Point) -> List[int]:
        """
        Résout le TSP pour un cluster avec LNS + recuit simulé + 2-opt.
        
        Args:
            points: Points à visiter
            depot: Point de départ/retour
            
        Returns:
            Liste des IDs de points dans l'ordre de visite optimal trouvé
        """
        if len(points) == 0:
            return []
        if len(points) == 1:
            return [points[0].id]
        
        # Solution initiale: plus proche voisin (comme spécifié)
        current = self._nearest_neighbor(points, depot)
        current_cost = self._cost(current, points, depot)
        
        # Amélioration locale 2-opt sur solution initiale
        current = self._two_opt(current, points, depot)
        current_cost = self._cost(current, points, depot)
        
        best = current[:]
        best_cost = current_cost
        
        # Itérations LNS avec recuit simulé
        for iteration in range(self.n_iter):
            # Température décroissante linéaire
            temp = self.temp_init * (1 - iteration / self.n_iter)
            
            # Phase de destruction: retirer aléatoirement des points
            n_remove = max(1, int(len(current) * self.destroy_rate))
            removed_indices = random.sample(current, n_remove)
            partial = [x for x in current if x not in removed_indices]
            
            # Phase de réparation: insertion gloutonne
            new_solution = self._greedy_repair(partial, removed_indices, points, depot)
            
            # Amélioration locale 2-opt
            new_solution = self._two_opt(new_solution, points, depot)
            new_cost = self._cost(new_solution, points, depot)
            
            # Critère d'acceptation (recuit simulé)
            delta = new_cost - current_cost
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                current = new_solution
                current_cost = new_cost
            
            # Mise à jour du meilleur
            if new_cost < best_cost:
                best = new_solution[:]
                best_cost = new_cost
        
        # Retourner la meilleure solution (IDs des points)
        return [points[i].id for i in best]
    
    def _nearest_neighbor(self, points: List[Point], depot: Point) -> List[int]:
        """Heuristique du plus proche voisin pour solution initiale"""
        n = len(points)
        unvisited = set(range(n))
        route = []
        
        # Commencer par le point le plus proche du dépôt
        if unvisited:
            current = min(unvisited, key=lambda i: depot.distance_to(points[i]))
        
        while unvisited:
            route.append(current)
            unvisited.remove(current)
            
            if unvisited:
                # Aller au plus proche voisin non visité
                current = min(unvisited, key=lambda i: points[current].distance_to(points[i]))
        
        return route
    
    def _cost(self, route: List[int], points: List[Point], depot: Point) -> float:
        """Calcule le coût (distance totale) d'une route"""
        if not route:
            return 0.0
        
        distance = depot.distance_to(points[route[0]])
        for i in range(len(route) - 1):
            distance += points[route[i]].distance_to(points[route[i + 1]])
        distance += points[route[-1]].distance_to(depot)
        
        return distance
    
    def _greedy_repair(self, partial: List[int], removed: List[int], 
                      points: List[Point], depot: Point) -> List[int]:
        """
        Réparation par insertion gloutonne.
        Insère chaque point retiré à la position qui minimise l'augmentation du coût.
        """
        route = partial[:]
        
        for node in removed:
            best_pos = 0
            best_cost = float('inf')
            
            # Essayer toutes les positions d'insertion
            for pos in range(len(route) + 1):
                test_route = route[:pos] + [node] + route[pos:]
                cost = self._cost(test_route, points, depot)
                
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            
            route.insert(best_pos, node)
        
        return route
    
    def _two_opt(self, route: List[int], points: List[Point], depot: Point) -> List[int]:
        """
        Amélioration locale 2-opt.
        Essaie d'inverser tous les segments de la route pour réduire les croisements.
        """
        if len(route) < 4:
            return route
        
        improved = True
        best_route = route[:]
        best_cost = self._cost(best_route, points, depot)
        
        while improved:
            improved = False
            
            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route)):
                    # Créer nouvelle route en inversant segment [i+1:j+1]
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    new_cost = self._cost(new_route, points, depot)
                    
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route


# =============================================================================
# SOLVEUR VRP - COMBINE K-MEANS ET TSP
# =============================================================================

class VRPSolver:
    """
    Résout le VRP en deux phases:
    1. Clustering K-means++ pour partitionner les points entre robots
    2. Résolution TSP pour chaque cluster (GA ou LNS)
    """
    
    def __init__(self, points: List[Point], n_robots: int):
        self.points = points
        self.n_robots = n_robots
        self.depot = Point(0, 0, 0)
    
    def solve_ga(self) -> Solution:
        """Résout avec K-means + GA"""
        start_time = time.time()
        
        # Phase 1: Clustering K-means++
        clusters = KMeansClustering.cluster(self.points, self.n_robots)
        
        # Phase 2: Résolution TSP avec GA pour chaque cluster
        ga = GeneticAlgorithm()
        distances = []
        points_dict = {p.id: p for p in self.points}
        
        for cluster in clusters:
            if not cluster:
                distances.append(0.0)
                continue
            
            route_ids = ga.solve(cluster, self.depot)
            distance = self._calculate_route_distance(route_ids, points_dict)
            distances.append(distance)
        
        elapsed_time = time.time() - start_time
        
        return self._create_solution('GA', distances, elapsed_time)
    
    def solve_lns(self) -> Solution:
        """Résout avec K-means + LNS"""
        start_time = time.time()
        
        # Phase 1: Clustering K-means++
        clusters = KMeansClustering.cluster(self.points, self.n_robots)
        
        # Phase 2: Résolution TSP avec LNS pour chaque cluster
        lns = LargeNeighborhoodSearch()
        distances = []
        points_dict = {p.id: p for p in self.points}
        
        for cluster in clusters:
            if not cluster:
                distances.append(0.0)
                continue
            
            route_ids = lns.solve(cluster, self.depot)
            distance = self._calculate_route_distance(route_ids, points_dict)
            distances.append(distance)
        
        elapsed_time = time.time() - start_time
        
        return self._create_solution('LNS', distances, elapsed_time)
    
    def _calculate_route_distance(self, route_ids: List[int], 
                                   points_dict: Dict[int, Point]) -> float:
        """Calcule la distance totale d'une route (incluant aller-retour dépôt)"""
        if not route_ids:
            return 0.0
        
        distance = self.depot.distance_to(points_dict[route_ids[0]])
        
        for i in range(len(route_ids) - 1):
            distance += points_dict[route_ids[i]].distance_to(points_dict[route_ids[i + 1]])
        
        distance += points_dict[route_ids[-1]].distance_to(self.depot)
        
        return distance
    
    def _create_solution(self, algo: str, distances: List[float], 
                        elapsed_time: float) -> Solution:
        """Crée un objet Solution avec toutes les métriques du document"""
        # Distance totale D
        total_distance = sum(distances)
        
        # Makespan T (robot le plus lent)
        max_distance = max(distances) if distances else 0.0
        
        # Coefficient de variation CV
        if len(distances) > 1 and np.mean(distances) > 0:
            cv = np.std(distances) / np.mean(distances)
        else:
            cv = 0.0
        
        return Solution(
            algo=algo,
            config='',
            n_points=len(self.points),
            n_robots=self.n_robots,
            field_type='',
            total_distance=total_distance,
            max_distance=max_distance,
            distances=distances,
            time_seconds=elapsed_time,
            cv=cv
        )


# =============================================================================
# BENCHMARK ET ANALYSE
# =============================================================================

class Benchmark:
    """Exécute le benchmark complet et génère tous les résultats"""
    
    def __init__(self):
        self.results: List[Solution] = []
        self.output_dir = 'resultats_benchmark'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run(self):
        """Exécute toutes les configurations du document"""
        print("\n" + "=" * 80)
        print(" BENCHMARK VRP AGRICOLE - GA vs LNS")
        print("=" * 80)
        print(f"Conformité stricte au document:")
        print(f"  • {CONFIG['RUNS_PER_CONFIG']} runs par configuration (page 45)")
        print(f"  • GA: pop={CONFIG['GA_POP_SIZE']}, gen={CONFIG['GA_N_GEN']} (page 33)")
        print(f"  • LNS: iter={CONFIG['LNS_N_ITER']}, destroy={CONFIG['LNS_DESTROY_RATE']} (page 33)")
        print("=" * 80 + "\n")
        
        field_types = ['rectangular', 'L_shaped', 'H_shaped']
        total_configs = len(CONFIG['N_POINTS']) * len(CONFIG['N_ROBOTS']) * len(field_types)
        config_count = 0
        
        for field_type in field_types:
            for n_points in CONFIG['N_POINTS']:
                for n_robots in CONFIG['N_ROBOTS']:
                    config_count += 1
                    print(f"\n[{config_count}/{total_configs}] Champ: {field_type:12s} | "
                          f"n={n_points:3d} | m={n_robots}")
                    
                    ga_solutions = []
                    lns_solutions = []
                    
                    for run in range(CONFIG['RUNS_PER_CONFIG']):
                        # Graine unique mais reproductible pour chaque run
                        seed = CONFIG['SEED_BASE'] + run * 1000 + \
                               hash(f"{field_type}{n_points}{n_robots}") % 10000
                        
                        # Générer les points
                        points = FieldGenerator.generate_points(field_type, n_points, seed)
                        
                        if len(points) < n_points * 0.7:
                            print(f"  Run {run+1}: Points insuffisants "
                                  f"({len(points)}/{n_points}), ignoré")
                            continue
                        
                        # Résoudre avec les deux approches
                        solver = VRPSolver(points, n_robots)
                        
                        ga_sol = solver.solve_ga()
                        ga_sol.config = f"{field_type}_{n_points}_{n_robots}"
                        ga_sol.field_type = field_type
                        ga_solutions.append(ga_sol)
                        
                        lns_sol = solver.solve_lns()
                        lns_sol.config = f"{field_type}_{n_points}_{n_robots}"
                        lns_sol.field_type = field_type
                        lns_solutions.append(lns_sol)
                        
                        print(f"  Run {run+1:2d}/{CONFIG['RUNS_PER_CONFIG']}: "
                              f"GA={ga_sol.total_distance:6.1f} ({ga_sol.time_seconds:.3f}s) | "
                              f"LNS={lns_sol.total_distance:6.1f} ({lns_sol.time_seconds:.3f}s)")
                    
                    # Agréger les résultats des runs
                    if ga_solutions and lns_solutions:
                        self._aggregate_and_store(ga_solutions, lns_solutions, 
                                                 field_type, n_points, n_robots)
    
    def _aggregate_and_store(self, ga_sols: List[Solution], lns_sols: List[Solution],
                            field_type: str, n_points: int, n_robots: int):
        """Agrège les résultats de plusieurs runs et calcule moyenne ± écart-type"""
        
        # Agréger GA
        ga_mean = Solution(
            algo='GA',
            config=f"{field_type}_{n_points}_{n_robots}",
            n_points=n_points,
            n_robots=n_robots,
            field_type=field_type,
            total_distance=np.mean([s.total_distance for s in ga_sols]),
            max_distance=np.mean([s.max_distance for s in ga_sols]),
            distances=[],  # On garde les stats agrégées
            time_seconds=np.mean([s.time_seconds for s in ga_sols]),
            cv=np.mean([s.cv for s in ga_sols])
        )
        
        # Agréger LNS
        lns_mean = Solution(
            algo='LNS',
            config=f"{field_type}_{n_points}_{n_robots}",
            n_points=n_points,
            n_robots=n_robots,
            field_type=field_type,
            total_distance=np.mean([s.total_distance for s in lns_sols]),
            max_distance=np.mean([s.max_distance for s in lns_sols]),
            distances=[],
            time_seconds=np.mean([s.time_seconds for s in lns_sols]),
            cv=np.mean([s.cv for s in lns_sols])
        )
        
        # Stocker les moyennes et écarts-types pour les graphiques
        ga_mean.distances = [
            np.std([s.total_distance for s in ga_sols]),
            np.std([s.max_distance for s in ga_sols])
        ]
        lns_mean.distances = [
            np.std([s.total_distance for s in lns_sols]),
            np.std([s.max_distance for s in lns_sols])
        ]
        
        self.results.extend([ga_mean, lns_mean])
    
    def print_summary(self):
        """Affiche le tableau récapitulatif dans la console"""
        print("\n" + "=" * 100)
        print(" TABLEAU RÉCAPITULATIF DES RÉSULTATS")
        print("=" * 100)
        
        # Grouper par configuration
        configs = {}
        for sol in self.results:
            key = (sol.field_type, sol.n_points, sol.n_robots)
            if key not in configs:
                configs[key] = {'GA': None, 'LNS': None}
            configs[key][sol.algo] = sol
        
        # Afficher par type de champ
        for field_type in ['rectangular', 'L_shaped', 'H_shaped']:
            print(f"\n### CHAMP: {field_type.upper().replace('_', ' ')}")
            print(f"{'n':>4} {'m':>4} | {'Algo':>4} | "
                  f"{'Distance':>12} {'Makespan':>12} {'Temps(s)':>10} {'CV':>8} | {'Gain(%)':>8}")
            print("-" * 80)
            
            for key in sorted(k for k in configs.keys() if k[0] == field_type):
                n, m = key[1], key[2]
                sols = configs[key]
                
                for algo in ['GA', 'LNS']:
                    sol = sols[algo]
                    if sol:
                        # Afficher moyenne ± écart-type
                        dist_str = f"{sol.total_distance:6.1f}±{sol.distances[0]:4.1f}"
                        make_str = f"{sol.max_distance:6.1f}±{sol.distances[1]:4.1f}"
                        
                        print(f"{n:>4} {m:>4} | {algo:>4} | "
                              f"{dist_str:>12} {make_str:>12} "
                              f"{sol.time_seconds:>10.3f} {sol.cv:>8.3f} | ", end="")
                        
                        if algo == 'LNS' and sols['GA']:
                            gain = (sols['GA'].total_distance - sol.total_distance) / \
                                   sols['GA'].total_distance * 100
                            print(f"{gain:>+7.1f}%")
                        else:
                            print("     ---")
        
        print("=" * 100)
    
    def save_text_report(self):
        """Sauvegarde un rapport texte détaillé"""
        filename = f"{self.output_dir}/rapport_resultats.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT DÉTAILLÉ - BENCHMARK VRP AGRICOLE\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Conformité: Paramètres exacts du document de recherche\n\n")
            
            f.write("PARAMÈTRES EXPÉRIMENTAUX:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Runs par configuration: {CONFIG['RUNS_PER_CONFIG']}\n")
            f.write(f"Points testés (n): {CONFIG['N_POINTS']}\n")
            f.write(f"Robots testés (m): {CONFIG['N_ROBOTS']}\n")
            f.write(f"Fonction de coût (z): {CONFIG['Z']}\n\n")
            
            f.write("PARAMÈTRES ALGORITHME GÉNÉTIQUE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Taille population: {CONFIG['GA_POP_SIZE']}\n")
            f.write(f"Nombre générations: {CONFIG['GA_N_GEN']}\n")
            f.write(f"Probabilité croisement: {CONFIG['GA_CROSSOVER_PROB']}\n")
            f.write(f"Probabilité mutation: {CONFIG['GA_MUTATION_PROB']}\n")
            f.write(f"Taille tournoi: {CONFIG['GA_TOURNAMENT_SIZE']}\n\n")
            
            f.write("PARAMÈTRES LARGE NEIGHBORHOOD SEARCH:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Nombre itérations: {CONFIG['LNS_N_ITER']}\n")
            f.write(f"Taux destruction: {CONFIG['LNS_DESTROY_RATE']}\n")
            f.write(f"Température initiale: {CONFIG['LNS_TEMP_INIT']}\n\n")
            
            f.write("RÉSULTATS PAR CONFIGURATION:\n")
            f.write("=" * 80 + "\n\n")
            
            # Grouper et écrire les résultats
            configs = {}
            for sol in self.results:
                key = (sol.field_type, sol.n_points, sol.n_robots)
                if key not in configs:
                    configs[key] = {'GA': None, 'LNS': None}
                configs[key][sol.algo] = sol
            
            for key in sorted(configs.keys()):
                field, n, m = key
                sols = configs[key]
                
                f.write(f"Configuration: {field} | n={n} | m={m}\n")
                f.write("-" * 80 + "\n")
                
                for algo in ['GA', 'LNS']:
                    sol = sols[algo]
                    if sol:
                        f.write(f"  {algo}:\n")
                        f.write(f"    Distance totale: {sol.total_distance:.2f} "
                               f"(± {sol.distances[0]:.2f})\n")
                        f.write(f"    Makespan:        {sol.max_distance:.2f} "
                               f"(± {sol.distances[1]:.2f})\n")
                        f.write(f"    Temps calcul:    {sol.time_seconds:.4f} s\n")
                        f.write(f"    CV:              {sol.cv:.4f}\n")
                
                if sols['GA'] and sols['LNS']:
                    gain_dist = (sols['GA'].total_distance - sols['LNS'].total_distance) / \
                               sols['GA'].total_distance * 100
                    gain_make = (sols['GA'].max_distance - sols['LNS'].max_distance) / \
                               sols['GA'].max_distance * 100
                    f.write(f"  → Gain LNS vs GA:\n")
                    f.write(f"    Distance: {gain_dist:+.2f}%\n")
                    f.write(f"    Makespan: {gain_make:+.2f}%\n")
                
                f.write("\n")
        
        print(f"\n Rapport sauvegardé: {filename}")
    
    def plot_all_results(self):
        """Génère tous les graphiques requis par le document"""
        print("\n Génération des graphiques...")
        
        # Extraire les données pour champ rectangulaire
        rect_data = [s for s in self.results if s.field_type == 'rectangular']
        
        # 1. Distance vs n (m=4, rect)
        self._plot_distance_vs_n(rect_data)
        
        # 2. Temps vs n (m=4, rect)
        self._plot_time_vs_n(rect_data)
        
        # 3. Makespan vs m (n=50, rect) - NOUVEAU comme dans le document
        self._plot_makespan_vs_m(rect_data)
        
        # 4. Gains LNS (toutes configs)
        self._plot_lns_gains()
        
        # 5. Comparaison formes (n=50, m=3)
        self._plot_field_comparison()
        
        # 6-8. Visualisations des champs
        self._visualize_fields()
        
        print(f" Tous les graphiques sauvegardés dans {self.output_dir}/")
    
    def _plot_distance_vs_n(self, data: List[Solution]):
        """Graphique 1: Distance totale vs nombre de points avec barres d'erreur"""
        plt.figure(figsize=(10, 6))
        
        ga_data = [(s.n_points, s.total_distance, s.distances[0]) 
                   for s in data if s.algo == 'GA' and s.n_robots == 4]
        lns_data = [(s.n_points, s.total_distance, s.distances[0]) 
                    for s in data if s.algo == 'LNS' and s.n_robots == 4]
        
        if ga_data:
            ga_data.sort()
            n_vals, dist_vals, err_vals = zip(*ga_data)
            plt.errorbar(n_vals, dist_vals, yerr=err_vals, marker='o', linewidth=2, 
                        markersize=8, capsize=5, label='K-means + GA', color='#2E86AB')
        
        if lns_data:
            lns_data.sort()
            n_vals, dist_vals, err_vals = zip(*lns_data)
            plt.errorbar(n_vals, dist_vals, yerr=err_vals, marker='s', linewidth=2, 
                        markersize=8, capsize=5, label='K-means + LNS', color='#A23B72')
        
        plt.xlabel('Nombre de points (n)', fontsize=12)
        plt.ylabel('Distance totale (unités)', fontsize=12)
        plt.title('Distance totale en fonction du nombre de points\n'
                 '(m=4 robots, champ rectangulaire)', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/1_distance_vs_points.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_vs_n(self, data: List[Solution]):
        """Graphique 2: Temps de calcul vs nombre de points"""
        plt.figure(figsize=(10, 6))
        
        ga_data = [(s.n_points, s.time_seconds) 
                   for s in data if s.algo == 'GA' and s.n_robots == 4]
        lns_data = [(s.n_points, s.time_seconds) 
                    for s in data if s.algo == 'LNS' and s.n_robots == 4]
        
        if ga_data:
            ga_data.sort()
            n_vals, time_vals = zip(*ga_data)
            plt.plot(n_vals, time_vals, 'o-', linewidth=2, markersize=8, 
                    label='K-means + GA', color='#2E86AB')
        
        if lns_data:
            lns_data.sort()
            n_vals, time_vals = zip(*lns_data)
            plt.plot(n_vals, time_vals, 's-', linewidth=2, markersize=8, 
                    label='K-means + LNS', color='#A23B72')
        
        plt.xlabel('Nombre de points (n)', fontsize=12)
        plt.ylabel('Temps de calcul (secondes)', fontsize=12)
        plt.title('Temps de calcul en fonction du nombre de points\n'
                 '(m=4 robots, champ rectangulaire)', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/2_temps_calcul.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_makespan_vs_m(self, data: List[Solution]):
        """Graphique 3: Makespan vs nombre de robots - NOUVEAU"""
        plt.figure(figsize=(10, 6))
        
        ga_data = [(s.n_robots, s.max_distance, s.distances[1]) 
                   for s in data if s.algo == 'GA' and s.n_points == 50]
        lns_data = [(s.n_robots, s.max_distance, s.distances[1]) 
                    for s in data if s.algo == 'LNS' and s.n_points == 50]
        
        if ga_data:
            ga_data.sort()
            m_vals, make_vals, err_vals = zip(*ga_data)
            plt.errorbar(m_vals, make_vals, yerr=err_vals, marker='o', linewidth=2, 
                        markersize=8, capsize=5, label='K-means + GA', color='#2E86AB')
        
        if lns_data:
            lns_data.sort()
            m_vals, make_vals, err_vals = zip(*lns_data)
            plt.errorbar(m_vals, make_vals, yerr=err_vals, marker='s', linewidth=2, 
                        markersize=8, capsize=5, label='K-means + LNS', color='#A23B72')
        
        plt.xlabel('Nombre de robots (m)', fontsize=12)
        plt.ylabel('Makespan (unités)', fontsize=12)
        plt.title('Makespan en fonction du nombre de robots\n'
                 '(n=50 points, champ rectangulaire)', fontsize=13, fontweight='bold')
        plt.xticks(CONFIG['N_ROBOTS'])
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/3_makespan_vs_robots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_lns_gains(self):
        """Graphique 4: Gains de LNS sur toutes les configurations"""
        plt.figure(figsize=(14, 6))
        
        configs = {}
        for sol in self.results:
            key = (sol.field_type, sol.n_points, sol.n_robots)
            if key not in configs:
                configs[key] = {'GA': None, 'LNS': None}
            configs[key][sol.algo] = sol
        
        labels = []
        gains = []
        
        for key in sorted(configs.keys()):
            field, n, m = key
            sols = configs[key]
            if sols['GA'] and sols['LNS']:
                gain = (sols['GA'].total_distance - sols['LNS'].total_distance) / \
                       sols['GA'].total_distance * 100
                labels.append(f"{field[:4]}\nn={n}\nm={m}")
                gains.append(gain)
        
        colors = ['#27AE60' if g > 0 else '#E74C3C' for g in gains]
        bars = plt.bar(range(len(gains)), gains, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=1)
        
        # Annoter les valeurs
        for i, (bar, gain) in enumerate(zip(bars, gains)):
            height = bar.get_height()
            y_pos = height + 1 if height > 0 else height - 1
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{gain:.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top', 
                    fontsize=8, fontweight='bold')
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
        plt.ylabel('Gain de LNS par rapport à GA (%)', fontsize=12)
        plt.title('Amélioration de la distance totale avec LNS\n'
                 '(toutes configurations)', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/4_gain_lns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_field_comparison(self):
        """Graphique 5: Comparaison des formes de champs"""
        plt.figure(figsize=(10, 6))
        
        field_data = {}
        for sol in self.results:
            if sol.n_points == 50 and sol.n_robots == 3:
                if sol.field_type not in field_data:
                    field_data[sol.field_type] = {'GA': None, 'LNS': None}
                field_data[sol.field_type][sol.algo] = sol
        
        field_names = ['rectangular', 'L_shaped', 'H_shaped']
        field_labels = ['Rectangulaire', 'En L', 'En H']
        
        x = np.arange(len(field_names))
        width = 0.35
        
        ga_vals = [field_data[f]['GA'].total_distance if f in field_data and field_data[f]['GA'] 
                   else 0 for f in field_names]
        lns_vals = [field_data[f]['LNS'].total_distance if f in field_data and field_data[f]['LNS'] 
                    else 0 for f in field_names]
        
        plt.bar(x - width/2, ga_vals, width, label='K-means + GA', 
               color='#2E86AB', alpha=0.7, edgecolor='black')
        plt.bar(x + width/2, lns_vals, width, label='K-means + LNS', 
               color='#A23B72', alpha=0.7, edgecolor='black')
        
        # Annoter les valeurs
        for i, (g, l) in enumerate(zip(ga_vals, lns_vals)):
            if g > 0:
                plt.text(i - width/2, g + 10, f'{g:.0f}', ha='center', 
                        va='bottom', fontsize=10, fontweight='bold')
            if l > 0:
                plt.text(i + width/2, l + 10, f'{l:.0f}', ha='center', 
                        va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Forme du champ', fontsize=12)
        plt.ylabel('Distance totale (unités)', fontsize=12)
        plt.title('Impact de la forme du champ\n(n=50 points, m=3 robots)', 
                 fontsize=13, fontweight='bold')
        plt.xticks(x, field_labels)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/5_comparaison_formes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_fields(self):
        """Graphiques 6-8: Visualisation des géométries des champs"""
        print("\n  Génération des visualisations des champs...")
        
        field_types = ['rectangular', 'L_shaped', 'H_shaped']
        field_titles = ['Rectangulaire', 'En forme de L', 'En forme de H']
        
        for field_type, title in zip(field_types, field_titles):
            # Générer 50 points pour visualisation
            points = FieldGenerator.generate_points(field_type, 50, CONFIG['SEED_BASE'])
            
            plt.figure(figsize=(8, 6))
            
            # Dessiner le contour du champ
            self._draw_field_boundary(field_type)
            
            # Points d'intérêt
            if points:
                xs = [p.x for p in points]
                ys = [p.y for p in points]
                plt.scatter(xs, ys, c='blue', s=40, alpha=0.6, 
                          label=f'{len(points)} points', zorder=3)
            
            # Dépôt (origine)
            plt.plot(0, 0, 'rs', markersize=14, label='Dépôt', zorder=4)
            
            plt.xlabel('X (unités)', fontsize=11)
            plt.ylabel('Y (unités)', fontsize=11)
            plt.title(f'Champ {title}', fontsize=13, fontweight='bold')
            plt.legend(fontsize=10, loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.xlim(-2, CONFIG['FIELD_WIDTH'] + 2)
            plt.ylim(-2, CONFIG['FIELD_HEIGHT'] + 2)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/champ_{field_type}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _draw_field_boundary(self, field_type: str):
        """Dessine le contour d'un champ selon sa géométrie"""
        w, h = CONFIG['FIELD_WIDTH'], CONFIG['FIELD_HEIGHT']
        
        if field_type == 'rectangular':
            plt.fill([0, w, w, 0], [0, 0, h, h], 'lightgray', alpha=0.3, zorder=1)
            plt.plot([0, w, w, 0, 0], [0, 0, h, h, 0], 'k-', linewidth=2, zorder=2)
        
        elif field_type == 'L_shaped':
            plt.fill([0, w, w, 20, 20, 0], [0, 0, 20, 20, h, h], 
                    'lightgray', alpha=0.3, zorder=1)
            plt.plot([0, w, w, 20, 20, 0, 0], [0, 0, 20, 20, h, h, 0], 
                    'k-', linewidth=2, zorder=2)
        
        elif field_type == 'H_shaped':
            # Trois rectangles pour former le H
            plt.fill([0, 18, 18, 0], [0, 0, h, h], 'lightgray', alpha=0.3, zorder=1)
            plt.fill([28, w, w, 28], [0, 0, h, h], 'lightgray', alpha=0.3, zorder=1)
            plt.fill([18, 28, 28, 18], [10, 10, 18, 18], 'lightgray', alpha=0.3, zorder=1)
            plt.plot([0, 18, 18, 28, 28, w, w, 28, 28, 18, 18, 0, 0],
                    [0, 0, 10, 10, 0, 0, h, h, 18, 18, h, h, 0], 
                    'k-', linewidth=2, zorder=2)


# =============================================================================
# MAIN - POINT D'ENTRÉE
# =============================================================================

def main():
    """Point d'entrée principal du programme"""
    print("\n" + "=" * 80)
    print(" BENCHMARK VRP AGRICOLE - Comparaison Rigoureuse GA vs LNS")
    print("=" * 80)
    print("Conformité stricte au document de recherche")
    print("Paramètres exacts - Section 4 : Implémentation et Expérimentation")
    print("=" * 80)
    
    # Créer et exécuter le benchmark
    benchmark = Benchmark()
    
    try:
        # Phase 1: Exécution des tests
        benchmark.run()
        
        # Phase 2: Affichage des résultats
        benchmark.print_summary()
        
        # Phase 3: Sauvegarde du rapport
        benchmark.save_text_report()
        
        # Phase 4: Génération des graphiques
        benchmark.plot_all_results()
        
        print("\n" + "=" * 80)
        print(" BENCHMARK TERMINÉ AVEC SUCCÈS!")
        print(f" Tous les fichiers sont dans: {benchmark.output_dir}/")
        print("=" * 80)
        print("\nFichiers générés:")
        print("  • rapport_resultats.txt")
        print("  • 1_distance_vs_points.png")
        print("  • 2_temps_calcul.png")
        print("  • 3_makespan_vs_robots.png")
        print("  • 4_gain_lns.png")
        print("  • 5_comparaison_formes.png")
        print("  • champ_rectangular.png")
        print("  • champ_L_shaped.png")
        print("  • champ_H_shaped.png")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

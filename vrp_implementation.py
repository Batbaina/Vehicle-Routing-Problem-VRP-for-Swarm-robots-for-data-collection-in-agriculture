"""
BENCHMARK SIMPLE - GA vs LNS pour VRP Agricole
================================================
Exécute les tests, affiche les résultats et sauvegarde les graphiques.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List, Tuple
import time
from dataclasses import dataclass
import random
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'FIELD_WIDTH': 46.0,
    'FIELD_HEIGHT': 28.0,
    
    # Configurations à tester
    'EXPERIMENTS': [
        # (field_type, n_points, n_robots, n_runs, description)
        ('rectangular', 15, 3, 3, "Petite instance"),
        ('rectangular', 30, 3, 3, "Instance moyenne"),
        ('rectangular', 50, 3, 3, "Grande instance"),
        ('L_shaped', 50, 3, 3, "Forme en L"),
        ('H_shaped', 50, 3, 3, "Forme en H"),
    ],
    
    'Z': 0.5,
}

# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class Point:
    x: float
    y: float
    id: int
    
    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Result:
    """Résultat d'une exécution"""
    algo: str
    total_distance: float
    max_distance: float
    time: float
    cv: float

# =============================================================================
# GÉNÉRATION DES POINTS
# =============================================================================

class FieldGenerator:
    @staticmethod
    def is_point_valid(x, y, field_type):
        w, h = CONFIG['FIELD_WIDTH'], CONFIG['FIELD_HEIGHT']
        
        if field_type == 'rectangular':
            return 0 <= x <= w and 0 <= y <= h
        elif field_type == 'L_shaped':
            return 0 <= x <= w and 0 <= y <= h and not (x > 20 and y > 20)
        elif field_type == 'H_shaped':
            if not (0 <= x <= w and 0 <= y <= h):
                return False
            # Forme en H: trois rectangles
            return (0 <= x <= 18) or (28 <= x <= w) or (10 <= y <= 18 and 18 <= x <= 28)
        return False
    
    @staticmethod
    def generate_points(field_type, n_points, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        points = []
        attempts = 0
        max_attempts = n_points * 100
        
        while len(points) < n_points and attempts < max_attempts:
            x = np.random.uniform(0, CONFIG['FIELD_WIDTH'])
            y = np.random.uniform(0, CONFIG['FIELD_HEIGHT'])
            if FieldGenerator.is_point_valid(x, y, field_type):
                points.append(Point(x, y, len(points) + 1))
            attempts += 1
        
        return points

# =============================================================================
# CLUSTERING
# =============================================================================

class KMeansCluster:
    @staticmethod
    def cluster(points, n_clusters):
        if len(points) <= n_clusters:
            clusters = [[] for _ in range(n_clusters)]
            for i, p in enumerate(points):
                clusters[i % n_clusters].append(p)
            return clusters
        
        X = np.array([[p.x, p.y] for p in points])
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        labels = kmeans.fit_predict(X)
        
        clusters = [[] for _ in range(n_clusters)]
        for i, p in enumerate(points):
            clusters[labels[i]].append(p)
        return clusters

# =============================================================================
# ALGORITHME GÉNÉTIQUE (simplifié)
# =============================================================================

class GeneticAlgorithm:
    def __init__(self):
        self.pop_size = 40
        self.n_gen = 60
    
    def solve(self, points, depot):
        if len(points) <= 1:
            return [p.id for p in points]
        
        n = len(points)
        # Population initiale
        pop = [list(range(n)) for _ in range(self.pop_size)]
        for p in pop:
            random.shuffle(p)
        
        best = None
        best_dist = float('inf')
        
        for _ in range(self.n_gen):
            # Évaluation
            distances = [self._distance(p, points, depot) for p in pop]
            
            # Meilleur
            idx = np.argmin(distances)
            if distances[idx] < best_dist:
                best_dist = distances[idx]
                best = pop[idx][:]
            
            # Sélection (garde les meilleurs)
            sorted_idx = np.argsort(distances)
            pop = [pop[i][:] for i in sorted_idx[:self.pop_size]]
            
            # Croisements
            new_pop = []
            for i in range(0, len(pop)-1, 2):
                if i+1 < len(pop):
                    c1, c2 = self._crossover(pop[i], pop[i+1])
                    new_pop.extend([c1, c2])
            
            pop = new_pop[:self.pop_size]
        
        return [points[i].id for i in best]
    
    def _distance(self, route, points, depot):
        if not route:
            return 0
        d = depot.distance_to(points[route[0]])
        for i in range(len(route)-1):
            d += points[route[i]].distance_to(points[route[i+1]])
        d += points[route[-1]].distance_to(depot)
        return d
    
    def _crossover(self, p1, p2):
        n = len(p1)
        if n < 2:
            return p1[:], p2[:]
        
        # Coupe simple au milieu
        cut = n // 2
        c1 = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
        c2 = p2[:cut] + [x for x in p1 if x not in p2[:cut]]
        return c1, c2

# =============================================================================
# LNS (simplifié)
# =============================================================================

class LargeNeighborhoodSearch:
    def __init__(self):
        self.n_iter = 50
        self.destroy_rate = 0.3
    
    def solve(self, points, depot):
        if len(points) <= 1:
            return [p.id for p in points]
        
        # Solution initiale
        current = self._nearest_neighbor(points, depot)
        current_cost = self._cost(current, points, depot)
        best = current[:]
        best_cost = current_cost
        
        for _ in range(self.n_iter):
            # Destruction
            n = max(1, int(len(current) * self.destroy_rate))
            removed = random.sample(current, n)
            partial = [x for x in current if x not in removed]
            
            # Réparation
            new = self._repair(partial, removed, points, depot)
            new_cost = self._cost(new, points, depot)
            
            if new_cost < best_cost:
                best = new[:]
                best_cost = new_cost
                current = new
        
        return [points[i].id for i in best]
    
    def _nearest_neighbor(self, points, depot):
        n = len(points)
        unvisited = set(range(n))
        route = []
        current = None
        
        while unvisited:
            if current is None:
                current = min(unvisited, key=lambda i: depot.distance_to(points[i]))
            else:
                current = min(unvisited, key=lambda i: points[current].distance_to(points[i]))
            route.append(current)
            unvisited.remove(current)
        return route
    
    def _cost(self, route, points, depot):
        if not route:
            return 0
        d = depot.distance_to(points[route[0]])
        for i in range(len(route)-1):
            d += points[route[i]].distance_to(points[route[i+1]])
        d += points[route[-1]].distance_to(depot)
        return d
    
    def _repair(self, partial, removed, points, depot):
        route = partial[:]
        for node in removed:
            # Insère à la meilleure position
            best_pos = 0
            best_cost = float('inf')
            for pos in range(len(route)+1):
                test = route[:pos] + [node] + route[pos:]
                cost = self._cost(test, points, depot)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            route.insert(best_pos, node)
        return route

# =============================================================================
# SOLVEUR
# =============================================================================

class VRPSolver:
    def __init__(self, points, n_robots):
        self.points = points
        self.n_robots = n_robots
        self.depot = Point(0, 0, 0)
    
    def solve_ga(self):
        start = time.time()
        clusters = KMeansCluster.cluster(self.points, self.n_robots)
        ga = GeneticAlgorithm()
        
        distances = []
        points_dict = {p.id: p for p in self.points}
        
        for cluster in clusters:
            if not cluster:
                distances.append(0)
                continue
            route = ga.solve(cluster, self.depot)
            if route:
                d = self.depot.distance_to(points_dict[route[0]])
                for i in range(len(route)-1):
                    d += points_dict[route[i]].distance_to(points_dict[route[i+1]])
                d += points_dict[route[-1]].distance_to(self.depot)
                distances.append(d)
            else:
                distances.append(0)
        
        total = sum(distances)
        max_dist = max(distances) if distances else 0
        cv = np.std(distances) / np.mean(distances) if len(distances) > 1 else 0
        
        return Result('GA', total, max_dist, time.time()-start, cv)
    
    def solve_lns(self):
        start = time.time()
        clusters = KMeansCluster.cluster(self.points, self.n_robots)
        lns = LargeNeighborhoodSearch()
        
        distances = []
        points_dict = {p.id: p for p in self.points}
        
        for cluster in clusters:
            if not cluster:
                distances.append(0)
                continue
            route = lns.solve(cluster, self.depot)
            if route:
                d = self.depot.distance_to(points_dict[route[0]])
                for i in range(len(route)-1):
                    d += points_dict[route[i]].distance_to(points_dict[route[i+1]])
                d += points_dict[route[-1]].distance_to(self.depot)
                distances.append(d)
            else:
                distances.append(0)
        
        total = sum(distances)
        max_dist = max(distances) if distances else 0
        cv = np.std(distances) / np.mean(distances) if len(distances) > 1 else 0
        
        return Result('LNS', total, max_dist, time.time()-start, cv)

# =============================================================================
# VISUALISATION DES CHAMPS
# =============================================================================

def plot_field(field_type, points=None, save_name=None):
    """Dessine la forme du champ et les points"""
    plt.figure(figsize=(8, 6))
    
    # Dessiner la forme du champ
    w, h = CONFIG['FIELD_WIDTH'], CONFIG['FIELD_HEIGHT']
    
    if field_type == 'rectangular':
        plt.fill([0, w, w, 0], [0, 0, h, h], 'lightgray', alpha=0.3)
        plt.plot([0, w, w, 0, 0], [0, 0, h, h, 0], 'k-', linewidth=2)
        
    elif field_type == 'L_shaped':
        # Rectangle avec coin manquant
        plt.fill([0, w, w, 20, 20, 0], 
                [0, 0, 20, 20, h, h], 'lightgray', alpha=0.3)
        plt.plot([0, w, w, 20, 20, 0, 0], 
                [0, 0, 20, 20, h, h, 0], 'k-', linewidth=2)
        
    elif field_type == 'H_shaped':
        # Barre gauche
        plt.fill([0, 18, 18, 0], [0, 0, h, h], 'lightgray', alpha=0.3)
        # Barre droite
        plt.fill([28, w, w, 28], [0, 0, h, h], 'lightgray', alpha=0.3)
        # Barre horizontale
        plt.fill([18, 28, 28, 18], [10, 10, 18, 18], 'lightgray', alpha=0.3)
        # Contour
        plt.plot([0, 18, 18, 28, 28, w, w, 28, 28, 18, 18, 0, 0],
                [0, 0, 10, 10, 0, 0, h, h, 18, 18, h, h, 0], 'k-', linewidth=2)
    
    # Dépôt
    plt.plot(0, 0, 'rs', markersize=12, label='Dépôt')
    
    # Points
    if points:
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        plt.scatter(xs, ys, c='blue', s=30, alpha=0.6, label=f'{len(points)} points')
    
    plt.xlabel('X (unités)')
    plt.ylabel('Y (unités)')
    plt.title(f'Champ {field_type.replace("_shaped", "").upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-2, w+2)
    plt.ylim(-2, h+2)
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# BENCHMARK PRINCIPAL
# =============================================================================

class SimpleBenchmark:
    def __init__(self):
        self.results = []  # Liste de (config, ga_result, lns_result)
        self.output_dir = 'resultats_benchmark'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run(self):
        """Exécute toutes les expériences"""
        print("\n" + "="*80)
        print("EXÉCUTION DU BENCHMARK GA vs LNS")
        print("="*80)
        
        total = len(CONFIG['EXPERIMENTS'])
        
        for i, (field, n, m, runs, desc) in enumerate(CONFIG['EXPERIMENTS']):
            print(f"\n [{i+1}/{total}] {desc} - {field}, n={n}, m={m}")
            
            ga_results = []
            lns_results = []
            
            for run in range(runs):
                print(f"   Run {run+1}/{runs}...", end=' ')
                
                seed = run * 100 + hash(f"{field}{n}{m}") % 10000
                points = FieldGenerator.generate_points(field, n, seed)
                
                if len(points) < n * 0.7:
                    print(" Pas assez de points")
                    continue
                
                solver = VRPSolver(points, m)
                
                ga = solver.solve_ga()
                lns = solver.solve_lns()
                
                ga_results.append(ga)
                lns_results.append(lns)
                
                print(f"GA={ga.total_distance:.1f} | LNS={lns.total_distance:.1f}")
            
            # Calculer les moyennes
            if ga_results:
                ga_mean = Result('GA', 
                               np.mean([r.total_distance for r in ga_results]),
                               np.mean([r.max_distance for r in ga_results]),
                               np.mean([r.time for r in ga_results]),
                               np.mean([r.cv for r in ga_results]))
                
                lns_mean = Result('LNS',
                                np.mean([r.total_distance for r in lns_results]),
                                np.mean([r.max_distance for r in lns_results]),
                                np.mean([r.time for r in lns_results]),
                                np.mean([r.cv for r in lns_results]))
                
                self.results.append({
                    'config': f"{field}_{n}_{m}",
                    'desc': desc,
                    'field': field,
                    'n': n,
                    'm': m,
                    'ga': ga_mean,
                    'lns': lns_mean
                })
    
    def print_table(self):
        """Affiche un tableau clair des résultats"""
        print("\n" + "="*100)
        print("RÉSULTATS DU BENCHMARK")
        print("="*100)
        
        # En-tête
        print(f"{'Configuration':<25} {'GA Distance':>15} {'LNS Distance':>15} {'Gain':>10} {'GA Temps':>12} {'LNS Temps':>12} {'GA CV':>10} {'LNS CV':>10}")
        print("-"*100)
        
        for r in self.results:
            ga = r['ga']
            lns = r['lns']
            gain = (ga.total_distance - lns.total_distance) / ga.total_distance * 100
            
            print(f"{r['config']:<25} "
                  f"{ga.total_distance:>8.1f} ± {np.std([ga.total_distance]):<4.1f} "
                  f"{lns.total_distance:>8.1f} ± {np.std([lns.total_distance]):<4.1f} "
                  f"{gain:>+9.1f}% "
                  f"{ga.time:>11.2f}s "
                  f"{lns.time:>11.2f}s "
                  f"{ga.cv:>9.3f} "
                  f"{lns.cv:>9.3f}")
        
        print("="*100)
    
    def plot_fields(self):
        """Dessine et sauvegarde les formes des champs"""
        print("\n Génération des formes de champs...")
        
        for field_type in ['rectangular', 'L_shaped', 'H_shaped']:
            points = FieldGenerator.generate_points(field_type, 50, seed=42)
            filename = f"{self.output_dir}/champ_{field_type}.png"
            plot_field(field_type, points, filename)
            print(f"   ✓ {filename}")
    
    def plot_comparison_charts(self):
        """Génère tous les graphiques de comparaison"""
        print("\n Génération des graphiques de comparaison...")
        
        # 1. Distance vs Nombre de points
        plt.figure(figsize=(10, 6))
        
        configs = [r for r in self.results if r['field'] == 'rectangular' and r['m'] == 3]
        configs.sort(key=lambda x: x['n'])
        
        if configs:
            n_vals = [c['n'] for c in configs]
            ga_vals = [c['ga'].total_distance for c in configs]
            lns_vals = [c['lns'].total_distance for c in configs]
            
            plt.plot(n_vals, ga_vals, 'o-', color='blue', linewidth=2, markersize=8, label='GA')
            plt.plot(n_vals, lns_vals, 's-', color='red', linewidth=2, markersize=8, label='LNS')
            
            plt.xlabel('Nombre de points')
            plt.ylabel('Distance totale')
            plt.title('Comparaison GA vs LNS - Distance totale')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(f"{self.output_dir}/1_distance_vs_points.png", dpi=300, bbox_inches='tight')
            print(f"   ✓ {self.output_dir}/1_distance_vs_points.png")
            plt.show()
        
        # 2. Temps de calcul
        plt.figure(figsize=(10, 6))
        
        if configs:
            ga_times = [c['ga'].time for c in configs]
            lns_times = [c['lns'].time for c in configs]
            
            plt.plot(n_vals, ga_times, 'o-', color='blue', linewidth=2, markersize=8, label='GA')
            plt.plot(n_vals, lns_times, 's-', color='red', linewidth=2, markersize=8, label='LNS')
            
            plt.xlabel('Nombre de points')
            plt.ylabel('Temps de calcul (secondes)')
            plt.title('Comparaison GA vs LNS - Temps d\'exécution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(f"{self.output_dir}/2_temps_calcul.png", dpi=300, bbox_inches='tight')
            print(f"   ✓ {self.output_dir}/2_temps_calcul.png")
            plt.show()
        
        # 3. Gain de LNS
        plt.figure(figsize=(12, 6))
        
        all_configs = [f"{r['config']}\n{r['desc']}" for r in self.results]
        gains = [(r['ga'].total_distance - r['lns'].total_distance) / r['ga'].total_distance * 100 
                for r in self.results]
        
        colors = ['green' if g > 0 else 'red' for g in gains]
        bars = plt.bar(range(len(gains)), gains, color=colors, alpha=0.7)
        
        for i, (bar, gain) in enumerate(zip(bars, gains)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{gain:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xticks(range(len(all_configs)), all_configs, rotation=45, ha='right')
        plt.ylabel('Gain de LNS par rapport à GA (%)')
        plt.title('Amélioration apportée par LNS')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/3_gain_lns.png", dpi=300, bbox_inches='tight')
        print(f"   ✓ {self.output_dir}/3_gain_lns.png")
        plt.show()
        
        # 4. Comparaison des formes (n=50, m=3)
        plt.figure(figsize=(10, 6))
        
        field_configs = [r for r in self.results if r['n'] == 50 and r['m'] == 3]
        
        if field_configs:
            fields = [r['field'].replace('_shaped', '') for r in field_configs]
            ga_vals = [r['ga'].total_distance for r in field_configs]
            lns_vals = [r['lns'].total_distance for r in field_configs]
            
            x = np.arange(len(fields))
            width = 0.35
            
            plt.bar(x - width/2, ga_vals, width, label='GA', color='blue', alpha=0.7)
            plt.bar(x + width/2, lns_vals, width, label='LNS', color='red', alpha=0.7)
            
            for i, (g, l) in enumerate(zip(ga_vals, lns_vals)):
                plt.text(i - width/2, g + 5, f'{g:.0f}', ha='center', va='bottom', fontsize=9)
                plt.text(i + width/2, l + 5, f'{l:.0f}', ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('Forme du champ')
            plt.ylabel('Distance totale')
            plt.title('Impact de la forme du champ (n=50, m=3)')
            plt.xticks(x, fields)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.savefig(f"{self.output_dir}/4_comparaison_formes.png", dpi=300, bbox_inches='tight')
            print(f"   ✓ {self.output_dir}/4_comparaison_formes.png")
            plt.show()
    
    def save_summary_text(self):
        """Sauvegarde un fichier texte récapitulatif"""
        filename = f"{self.output_dir}/resultats.txt"
        
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RÉSULTATS DU BENCHMARK GA vs LNS\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for r in self.results:
                f.write(f"\n--- {r['config']} - {r['desc']} ---\n")
                f.write(f"GA  : Distance={r['ga'].total_distance:.1f}, "
                       f"Temps={r['ga'].time:.2f}s, CV={r['ga'].cv:.3f}\n")
                f.write(f"LNS : Distance={r['lns'].total_distance:.1f}, "
                       f"Temps={r['lns'].time:.2f}s, CV={r['lns'].cv:.3f}\n")
                gain = (r['ga'].total_distance - r['lns'].total_distance) / r['ga'].total_distance * 100
                f.write(f"Gain LNS/GA: {gain:+.1f}%\n")
        
        print(f"\n Résumé sauvegardé dans {filename}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print(" BENCHMARK VRP - GA vs LNS")
    print("="*80)
    
    # Créer le benchmark
    bench = SimpleBenchmark()
    
    # Exécuter les tests
    bench.run()
    
    # Afficher les résultats
    bench.print_table()
    
    # Sauvegarder le résumé texte
    bench.save_summary_text()
    
    # Générer les visualisations
    print("\n Génération des visualisations...")
    bench.plot_fields()
    bench.plot_comparison_charts()
    
    print("\n" + "="*80)
    print("TERMINÉ!")
    print(f" Tous les fichiers sont dans le dossier: {bench.output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()

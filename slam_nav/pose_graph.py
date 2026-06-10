import math
import numpy as np
from slam_nav import math_utils as mu


class PoseGraph:

    def __init__(self, backend='auto'):
        if backend == 'auto':
            backend = 'g2o' if _HAS_G2O else 'gauss_newton'
        if backend == 'g2o' and not _HAS_G2O:
            raise RuntimeError(
                "backend='g2o' requested but g2o-python is not installed")
        if backend not in ('g2o', 'gauss_newton'):
            raise ValueError(f'unknown backend: {backend}')
        self.backend = backend

        self.nodes = {}        
        self.edges = []       
        self.fixed_ids =set()

   
    def add_node(self, node_id, x, y, theta, fixed=False):
       
        self.nodes[node_id] = np.array(
            [float(x), float(y), mu.normalize_angle(float(theta))])
        if fixed:
            self.fixed_ids.add(node_id)

    def add_edge(self, from_id, to_id, dx, dy, dtheta, information=None):
       
        if information is None:
            information = np.eye(3)
        self.edges.append({
            'from': from_id,
            'to': to_id,
            'z': np.array(
                [float(dx), float(dy), mu.normalize_angle(float(dtheta))]),
            'info': np.asarray(information, dtype=float),
        })

    
    def get_pose(self, node_id):
        return tuple(self.nodes[node_id])

    def get_all_poses(self):
        return {nid: tuple(p) for nid, p in self.nodes.items()}

    def num_nodes(self):
        return len(self.nodes)

    def num_edges(self):
        return len(self.edges)

    def chi2(self):
       
        total = 0.0
        for e in self.edges:
            err = self._edge_error(e)
            total += float(err @ e['info'] @ err)
        return total

    def optimize(self, iterations=10, tolerance=1e-6):
        
        if not self.nodes or not self.edges:
            return self.chi2()
        if not self.fixed_ids:
           
            self.fixed_ids.add(min(self.nodes.keys()))
        if self.backend == 'g2o':
            return self._optimize_g2o(iterations)
        return self._optimize_gauss_newton(iterations, tolerance)

    def _optimize_gauss_newton(self, max_iters, tol):
        node_ids = sorted(self.nodes.keys())
        idx = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)

        prev_chi2 = self.chi2()

        for _ in range(max_iters):
            H = np.zeros((3 * N, 3 * N))
            g = np.zeros(3 * N)

            for edge in self.edges:
                i = idx[edge['from']]
                j = idx[edge['to']]
                xi = self.nodes[edge['from']]
                xj = self.nodes[edge['to']]
                err, Ji, Jj = self._edge_error_and_jacobians(xi, xj, edge['z'])
                Omega = edge['info']

                ii = slice(3 * i, 3 * i + 3)
                jj = slice(3 * j, 3 * j + 3)
                H[ii, ii] += Ji.T @ Omega @ Ji
                H[ii, jj] += Ji.T @ Omega @ Jj
                H[jj, ii] += Jj.T @ Omega @ Ji
                H[jj, jj] += Jj.T @ Omega @ Jj
                g[ii] += Ji.T @ Omega @ err
                g[jj]+= Jj.T @ Omega @ err

  
            for nid in self.fixed_ids:
                k = idx[nid]
                H[3*k:3*k+3, 3*k:3*k+3] += 1e10 * np.eye(3)
                g[3*k:3*k+3] = 0.0

            H += 1e-6 * np.eye(3 * N)

            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break

           
            for k, nid in enumerate(node_ids):
                if nid in self.fixed_ids:
                    continue
                self.nodes[nid][0] += dx[3 * k]
                self.nodes[nid][1] += dx[3 * k + 1]
                self.nodes[nid][2] = mu.normalize_angle(
                    self.nodes[nid][2] + dx[3 * k + 2])

            chi2 = self.chi2()
            if abs(prev_chi2 - chi2) < tol:
                break
            prev_chi2 = chi2

        return self.chi2()

    @staticmethod
    def _edge_error_and_jacobians(xi, xj, z):
        ti = xi[2]
        c, s = math.cos(ti), math.sin(ti)
        dx = xj[0] -xi[0]
        dy = xj[1]- xi[1]

        h = np.array([
            c * dx + s * dy,
            -s * dx + c * dy,
            mu.angle_diff(xj[2], ti),
        ])
        err = z - h
        err[2] = mu.angle_diff(z[2], h[2])

  
        Ji = np.array([
            [ c,  s,  s * dx - c * dy],
            [-s,  c,  c * dx + s * dy],
            [ 0,  0,  1.0],
        ])
        Jj = np.array([
            [-c, -s, 0.0],
            [ s, -c, 0.0],
            [ 0,  0, -1.0],
        ])
        return err, Ji, Jj

    def _edge_error(self, edge):
        xi = self.nodes[edge['from']]
        xj = self.nodes[edge['to']]
        err, _, _ = self._edge_error_and_jacobians(xi, xj, edge['z'])
        return err

   
    def _optimize_g2o(self, iterations):
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE2(g2o.LinearSolverEigenSE2())
        algo = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(algo)

        for nid, pose in self.nodes.items():
            v = g2o.VertexSE2()
            v.set_id(int(nid))
            v.set_estimate(g2o.SE2(pose[0], pose[1], pose[2]))
            if nid in self.fixed_ids:
                v.set_fixed(True)
            opt.add_vertex(v)

        for edge in self.edges:
            e = g2o.EdgeSE2()
            e.set_vertex(0, opt.vertex(int(edge['from'])))
            e.set_vertex(1, opt.vertex(int(edge['to'])))
            e.set_measurement(
                g2o.SE2(edge['z'][0], edge['z'][1], edge['z'][2]))
            e.set_information(edge['info'])
            opt.add_edge(e)

        opt.initialize_optimization()
        opt.optimize(iterations)

        for nid in self.nodes:
            est = opt.vertex(int(nid)).estimate()
  
            try:
                tr = est.translation()
                ang = est.rotation().angle()
                self.nodes[nid] = np.array([tr[0], tr[1], ang])
            except AttributeError:
                vec = est.to_vector()
                self.nodes[nid] = np.array([vec[0], vec[1], vec[2]])

        return self.chi2()

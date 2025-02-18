from ortools.sat.python import cp_model
from qubots.base_optimizer import BaseOptimizer
from collections import defaultdict
import random

class ConstraintProgrammingBatchOptimizer(BaseOptimizer):
    """
    Advanced constraint programming optimizer for batch scheduling
    Uses OR-Tools CP-SAT solver with custom search strategies
    """
    
    def __init__(self, time_limit=300, num_search_workers=4):
        self.time_limit = time_limit
        self.num_search_workers = num_search_workers

    def optimize(self, problem, initial_solution=None, **kwargs):
        model = cp_model.CpModel()
        
        # Create main variables
        task_vars = self._create_variables(model, problem)
        batch_vars = self._create_batch_variables(model, problem, task_vars)
        
        # Add constraints
        self._add_core_constraints(model, problem, task_vars, batch_vars)
        self._add_precedence_constraints(model, problem, task_vars)
        
        # Set objective
        makespan = model.NewIntVar(0, problem.time_horizon, 'makespan')
        model.AddMaxEquality(makespan, [task_vars[t]['end'] for t in range(problem.nb_tasks)])
        model.Minimize(makespan)
        
        # Solve model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = self.num_search_workers
        status = solver.Solve(model)
        
        # Extract solution
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._extract_solution(solver, problem, task_vars, batch_vars)
        else:
            return problem.random_solution(), float('inf')

    def _create_variables(self, model, problem):
        """Create task-level decision variables"""
        task_vars = []
        for t in range(problem.nb_tasks):
            var = {
                'start': model.NewIntVar(0, problem.time_horizon, f'task{t}_start'),
                'end': model.NewIntVar(0, problem.time_horizon, f'task{t}_end'),
                'batch': model.NewIntVar(0, problem.nb_tasks, f'task{t}_batch'),
                'type': problem.types[t],
                'resource': problem.resources[t]
            }
            model.Add(var['end'] == var['start'] + problem.duration[t])
            task_vars.append(var)
        return task_vars

    def _create_batch_variables(self, model, problem, task_vars):
        """Create batch-level variables and constraints"""
        batch_groups = defaultdict(list)
        batch_vars = []
        
        # Group tasks by resource and type
        for t, var in enumerate(task_vars):
            key = (var['resource'], var['type'])
            batch_groups[key].append(t)
        
        # Create batch variables
        for (resource, type_), tasks in batch_groups.items():
            capacity = problem.capacity[resource]
            durations = [problem.duration[t] for t in tasks]
            
            # Create batch assignments
            batch_count = len(tasks) // capacity + 1
            for b in range(batch_count):
                batch_var = {
                    'start': model.NewIntVar(0, problem.time_horizon, f'batch_{resource}_{type_}_{b}_start'),
                    'end': model.NewIntVar(0, problem.time_horizon, f'batch_{resource}_{type_}_{b}_end'),
                    'tasks': []
                }
                batch_vars.append(batch_var)
                
                # Link tasks to batches
                for i in range(capacity):
                    if b*capacity + i < len(tasks):
                        t = tasks[b*capacity + i]
                        model.Add(task_vars[t]['batch'] == b)
                        model.Add(task_vars[t]['start'] == batch_var['start'])
                        model.Add(task_vars[t]['end'] == batch_var['end'])
        
        return batch_vars

    def _add_core_constraints(self, model, problem, task_vars, batch_vars):
        """Add capacity and resource constraints"""
        # Resource non-overlap
        for resource in range(problem.nb_resources):
            intervals = []
            for var in task_vars:
                if var['resource'] == resource:
                    intervals.append(var['start'])
                    intervals.append(var['end'])
            model.AddNoOverlap(intervals)
            
        # Batch capacity constraints
        for batch in batch_vars:
            model.Add(sum(1 for t in batch['tasks']) <= problem.capacity[batch['resource']])

    def _add_precedence_constraints(self, model, problem, task_vars):
        """Handle task precedence relationships"""
        for t in range(problem.nb_tasks):
            for s in problem.successors[t]:
                model.Add(task_vars[t]['end'] <= task_vars[s]['start'])

    def _extract_solution(self, solver, problem, task_vars, batch_vars):
        """Convert solver output to qubots format"""
        batch_schedule = []
        
        # Group tasks by their batch assignments
        batches = defaultdict(list)
        for t, var in enumerate(task_vars):
            batch_key = (var['resource'], solver.Value(var['batch']))
            batches[batch_key].append(t)
        
        # Create batch entries
        for (resource, batch_id), tasks in batches.items():
            start = min(solver.Value(task_vars[t]['start']) for t in tasks)
            end = max(solver.Value(task_vars[t]['end']) for t in tasks)
            
            batch_schedule.append({
                'resource': resource,
                'tasks': tasks,
                'start': start,
                'end': end
            })
        
        return {'batch_schedule': batch_schedule}, solver.ObjectiveValue()
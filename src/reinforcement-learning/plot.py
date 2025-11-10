# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import sparse
import imageio
import os
import tempfile
import copy

# %%
def build_exact_cover(n=2):
    N = n * n
    block = n

    rows = N * N * N
    cols = 4 * N * N

    idx = np.indices((N, N, N), dtype=np.int32)
    r, c, d = idx[0].ravel(), idx[1].ravel(), idx[2].ravel()

    col_cell    = r * N + c
    col_rownum  = N*N + r * N + d
    col_colnum  = 2*N*N + c * N + d
    box         = (r // block) * block + (c // block)
    col_boxnum  = 3*N*N + box * N + d

    nnz = rows * 4
    row_indices = np.repeat(np.arange(rows, dtype=np.int32), 4)
    col_indices = np.empty(nnz, dtype=np.int32)
    col_indices[0::4] = col_cell
    col_indices[1::4] = col_rownum
    col_indices[2::4] = col_colnum
    col_indices[3::4] = col_boxnum
    data = np.ones(nnz, dtype=np.int8)

    M = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(rows, cols)).tocsr()
    return M, N

# %%
def plot_sparsity(M, N):
    M_dense = M.toarray()
    
    colors = ['#808080', '#000000']
    cmap = ListedColormap(colors)
    
    N2 = N * N
    boundaries = [N2, 2*N2, 3*N2, 4*N2]
    labels = ['Cell', 'Row-Number', 'Column-Number', 'Box-Number']
    
    plt.figure(figsize=(12, 10))
    plt.imshow(M_dense, cmap=cmap, aspect='auto', interpolation='nearest')
    
    for i, boundary in enumerate(boundaries[:-1]):
        plt.axvline(x=boundary - 0.5, color='red', linewidth=1.5, linestyle='--')
    
    for i, (start, end, label) in enumerate(zip([0] + boundaries[:-1], boundaries, labels)):
        mid = (start + end) / 2
        plt.text(mid, -M.shape[0] * 0.05, label, ha='center', va='top', 
                fontsize=10, fontweight='bold')
    
    plt.xlabel(f"Constraint columns (0 .. {M.shape[1]-1})")
    plt.ylabel(f"Assignment rows (0 .. {M.shape[0]-1})")
    plt.title(f"Exact-cover matrix for Sudoku {N}×{N} (order {int(np.sqrt(N))})")
    plt.tight_layout()
    plt.show()

# %%
def plot_with_choice(M, N, choices):
    if isinstance(choices, tuple) and len(choices) == 3:
        choices = [choices]
    
    for x, y, value in choices:
        if not (0 <= x < N and 0 <= y < N and 0 <= value < N):
            raise ValueError(f"Invalid input: x, y, and value must be in range [0, {N-1}]. Got x={x}, y={y}, value={value}")
    
    M_dense = M.toarray()
    
    chosen_rows = []
    for x, y, value in choices:
        chosen_row = x * N * N + y * N + value
        chosen_rows.append(chosen_row)
    chosen_rows = set(chosen_rows)
    
    all_chosen_cols = set()
    for chosen_row in chosen_rows:
        chosen_cols = np.where(M_dense[chosen_row, :] == 1)[0]
        all_chosen_cols.update(chosen_cols)
    
    conflicting_rows = set()
    for col in all_chosen_cols:
        rows_with_one = np.where(M_dense[:, col] == 1)[0]
        conflicting_rows.update(rows_with_one)
    conflicting_rows -= chosen_rows
    conflicting_rows = list(conflicting_rows)
    
    highlight_matrix = M_dense.copy().astype(float)
    
    for row in conflicting_rows:
        highlight_matrix[row, M_dense[row, :] == 0] = 5
        highlight_matrix[row, M_dense[row, :] == 1] = 2
    
    for row in conflicting_rows:
        conflict_cols = np.intersect1d(
            list(all_chosen_cols),
            np.where(M_dense[row, :] == 1)[0]
        )
        highlight_matrix[row, conflict_cols] = 4
    
    for chosen_row in chosen_rows:
        highlight_matrix[chosen_row, M_dense[chosen_row, :] == 0] = 6
        highlight_matrix[chosen_row, M_dense[chosen_row, :] == 1] = 3
    
    for col in all_chosen_cols:
        for row in range(M_dense.shape[0]):
            if row not in chosen_rows and row not in conflicting_rows:
                if highlight_matrix[row, col] == 0:
                    highlight_matrix[row, col] = 6
    
    colors = ['#808080', '#000000', '#000000', '#0000FF', '#FF0000', '#FFB6C1', '#9999FF']
    cmap = ListedColormap(colors)
    
    N2 = N * N
    boundaries = [N2, 2*N2, 3*N2, 4*N2]
    labels = ['Cell', 'Row-Number', 'Column-Number', 'Box-Number']
    
    plt.figure(figsize=(12, 10))
    plt.imshow(highlight_matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=6)
    
    for i, boundary in enumerate(boundaries[:-1]):
        plt.axvline(x=boundary - 0.5, color='red', linewidth=1.5, linestyle='--')
    
    for i, (start, end, label) in enumerate(zip([0] + boundaries[:-1], boundaries, labels)):
        mid = (start + end) / 2
        plt.text(mid, -M.shape[0] * 0.05, label, ha='center', va='top', 
                fontsize=10, fontweight='bold')
    
    for chosen_row in chosen_rows:
        plt.axhline(y=chosen_row - 0.5, color='lightblue', linewidth=2, linestyle='-', alpha=0.7)
    
    choices_str = ', '.join([f"(r={x}, c={y}, d={v})" for x, y, v in choices])
    
    plt.xlabel(f"Constraint columns (0 .. {M.shape[1]-1})")
    plt.ylabel(f"Assignment rows (0 .. {M.shape[0]-1})")
    plt.title(f"Exact-cover matrix: Choices {choices_str} - Blue=chosen, Red=conflicting 1s, Pink=conflicting 0s, Orange=conflict cells")
    plt.tight_layout()
    plt.show()
    
    print(f"Chosen rows: {sorted(chosen_rows)}")
    for x, y, value in choices:
        chosen_row = x * N * N + y * N + value
        print(f"  Row {chosen_row}: (r={x}, c={y}, d={value})")
    print(f"Number of conflicting rows: {len(conflicting_rows)}")

# %%
def plot_sudoku(N, choices):
    grid = np.full((N, N), -1, dtype=int)
    
    for x, y, value in choices:
        if 0 <= x < N and 0 <= y < N and 0 <= value < N:
            grid[x, y] = value
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for i in range(N + 1):
        linewidth = 3 if i % int(np.sqrt(N)) == 0 else 1
        ax.axhline(i, color='black', linewidth=linewidth)
        ax.axvline(i, color='black', linewidth=linewidth)
    
    for i in range(N):
        for j in range(N):
            if grid[i, j] >= 0:
                ax.add_patch(plt.Rectangle((j, N-1-i), 1, 1, 
                                         facecolor='lightblue', 
                                         edgecolor='black', 
                                         linewidth=1))
                ax.text(j + 0.5, N-1-i + 0.5, str(grid[i, j] + 1), 
                       ha='center', va='center', fontsize=16, fontweight='bold')
            else:
                ax.add_patch(plt.Rectangle((j, N-1-i), 1, 1, 
                                         facecolor='white', 
                                         edgecolor='black', 
                                         linewidth=1))
    
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Sudoku {N}×{N}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"Grid filled with {len(choices)} values")

# %%
def save_plot_with_choice(M, N, choices, filename):
    if isinstance(choices, tuple) and len(choices) == 3:
        choices = [choices]
    
    for x, y, value in choices:
        if not (0 <= x < N and 0 <= y < N and 0 <= value < N):
            raise ValueError(f"Invalid input: x, y, and value must be in range [0, {N-1}]. Got x={x}, y={y}, value={value}")
    
    M_dense = M.toarray()
    
    chosen_rows = []
    for x, y, value in choices:
        chosen_row = x * N * N + y * N + value
        chosen_rows.append(chosen_row)
    chosen_rows = set(chosen_rows)
    
    all_chosen_cols = set()
    for chosen_row in chosen_rows:
        chosen_cols = np.where(M_dense[chosen_row, :] == 1)[0]
        all_chosen_cols.update(chosen_cols)
    
    conflicting_rows = set()
    for col in all_chosen_cols:
        rows_with_one = np.where(M_dense[:, col] == 1)[0]
        conflicting_rows.update(rows_with_one)
    conflicting_rows -= chosen_rows
    conflicting_rows = list(conflicting_rows)
    
    highlight_matrix = M_dense.copy().astype(float)
    
    for row in conflicting_rows:
        highlight_matrix[row, M_dense[row, :] == 0] = 5
        highlight_matrix[row, M_dense[row, :] == 1] = 2
    
    for row in conflicting_rows:
        conflict_cols = np.intersect1d(
            list(all_chosen_cols),
            np.where(M_dense[row, :] == 1)[0]
        )
        highlight_matrix[row, conflict_cols] = 4
    
    for chosen_row in chosen_rows:
        highlight_matrix[chosen_row, M_dense[chosen_row, :] == 0] = 6
        highlight_matrix[chosen_row, M_dense[chosen_row, :] == 1] = 3
    
    for col in all_chosen_cols:
        for row in range(M_dense.shape[0]):
            if row not in chosen_rows and row not in conflicting_rows:
                if highlight_matrix[row, col] == 0:
                    highlight_matrix[row, col] = 6
    
    colors = ['#808080', '#000000', '#000000', '#0000FF', '#FF0000', '#FFB6C1', '#ADD8E6']
    cmap = ListedColormap(colors)
    
    N2 = N * N
    boundaries = [N2, 2*N2, 3*N2, 4*N2]
    labels = ['Cell', 'Row-Number', 'Column-Number', 'Box-Number']
    
    plt.figure(figsize=(12, 10))
    plt.imshow(highlight_matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=6)
    
    for i, boundary in enumerate(boundaries[:-1]):
        plt.axvline(x=boundary - 0.5, color='red', linewidth=1.5, linestyle='--')
    
    for i, (start, end, label) in enumerate(zip([0] + boundaries[:-1], boundaries, labels)):
        mid = (start + end) / 2
        plt.text(mid, -M.shape[0] * 0.05, label, ha='center', va='top', 
                fontsize=10, fontweight='bold')
    
    for chosen_row in chosen_rows:
        plt.axhline(y=chosen_row - 0.5, color='lightblue', linewidth=2, linestyle='-', alpha=0.7)
    
    choices_str = ', '.join([f"(r={x}, c={y}, d={v})" for x, y, v in choices])
    
    plt.xlabel(f"Constraint columns (0 .. {M.shape[1]-1})")
    plt.ylabel(f"Assignment rows (0 .. {M.shape[0]-1})")
    # plt.title(f"Exact-cover matrix: Choices {choices_str} - Blue=chosen, Red=conflicting 1s, Pink=conflicting 0s, Orange=conflict cells")
    plt.xlim(-0.5, M.shape[1] - 0.5)
    plt.ylim(M.shape[0] - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# %%
def algorithm_x_with_gif(n, givens, output_gif='algorithm_x.gif'):
    M, N = build_exact_cover(n)
    M_dense = M.toarray()
    
    given_rows = set()
    for x, y, value in givens:
        given_row = x * N * N + y * N + value
        given_rows.add(given_row)
    
    covered_cols = set()
    for given_row in given_rows:
        cols = np.where(M_dense[given_row, :] == 1)[0]
        covered_cols.update(cols)
    
    conflicting_rows = set()
    for col in covered_cols:
        rows_with_one = np.where(M_dense[:, col] == 1)[0]
        conflicting_rows.update(rows_with_one)
    conflicting_rows -= given_rows
    
    active_rows = set(range(M_dense.shape[0])) - conflicting_rows - given_rows
    active_cols = set(range(M_dense.shape[1])) - covered_cols
    
    temp_dir = tempfile.mkdtemp()
    image_files = []
    frame_count = 0
    
    solution = []
    for x, y, value in givens:
        given_row = x * N * N + y * N + value
        solution.append(given_row)
    
    def solve(matrix, rows, cols, sol, depth=0):
        nonlocal frame_count
        
        if len(cols) == 0:
            final_choices = []
            for row in sol:
                x = row // (N * N)
                y = (row // N) % N
                value = row % N
                final_choices.append((x, y, value))
            save_plot_with_choice(M, N, final_choices, os.path.join(temp_dir, f'frame_{frame_count:04d}.png'))
            image_files.append(os.path.join(temp_dir, f'frame_{frame_count:04d}.png'))
            frame_count += 1
            return sol
        
        if len(rows) == 0:
            return None
        
        col = min(cols, key=lambda c: sum(1 for r in rows if matrix[r, c] == 1))
        
        rows_with_col = [r for r in rows if matrix[r, col] == 1]
        
        for row in rows_with_col:
            new_sol = sol + [row]
            
            current_choices = []
            for r in new_sol:
                x = r // (N * N)
                y = (r // N) % N
                v = r % N
                current_choices.append((x, y, v))
            
            save_plot_with_choice(M, N, current_choices, os.path.join(temp_dir, f'frame_{frame_count:04d}.png'))
            image_files.append(os.path.join(temp_dir, f'frame_{frame_count:04d}.png'))
            frame_count += 1
            
            new_rows = rows.copy()
            new_cols = cols.copy()
            
            cols_to_remove = set(np.where(matrix[row, :] == 1)[0])
            new_cols -= cols_to_remove
            
            rows_to_remove = set()
            for c in cols_to_remove:
                rows_with_one = np.where(matrix[:, c] == 1)[0]
                rows_to_remove.update(rows_with_one)
            new_rows -= rows_to_remove
            
            result = solve(matrix, new_rows, new_cols, new_sol, depth + 1)
            if result is not None:
                return result
        
        return None
    
    result = solve(M_dense, active_rows, active_cols, solution)
    
    if result is None:
        print("No solution found")
        return None
    
    final_choices = []
    for row in result:
        x = row // (N * N)
        y = (row // N) % N
        value = row % N
        final_choices.append((x, y, value))
    
    if len(image_files) > 0:
        from PIL import Image
        images = []
        target_size = None
        for f in image_files:
            img = Image.open(f)
            if target_size is None:
                target_size = img.size
            else:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(np.array(img))
        imageio.mimsave(output_gif, images, duration=0.1)
        print(f"GIF saved to {output_gif}")
    
    for f in image_files:
        os.remove(f)
    os.rmdir(temp_dir)
    
    return final_choices

# %%
class DLXNode:
    def __init__(self, row=-1, col=-1):
        self.row = row
        self.col = col
        self.up = self
        self.down = self
        self.left = self
        self.right = self
        self.header = None
        self.size = 0

# %%
def build_dlx_structure(M, N):
    M_dense = M.toarray()
    rows, cols = M_dense.shape
    
    root = DLXNode()
    headers = [DLXNode(-1, c) for c in range(cols)]
    
    for i, header in enumerate(headers):
        header.header = header
        header.size = int(np.sum(M_dense[:, i]))
        if i == 0:
            header.left = root
            root.right = header
        else:
            header.left = headers[i-1]
            headers[i-1].right = header
        if i == len(headers) - 1:
            header.right = root
            root.left = header
    
    nodes_by_row = {}
    last_node_in_col = {c: headers[c] for c in range(cols)}
    
    for r in range(rows):
        row_nodes = []
        for c in range(cols):
            if M_dense[r, c] == 1:
                node = DLXNode(r, c)
                node.header = headers[c]
                row_nodes.append(node)
                
                last_node = last_node_in_col[c]
                last_node.down = node
                node.up = last_node
                node.down = headers[c]
                headers[c].up = node
                last_node_in_col[c] = node
        if row_nodes:
            for i, node in enumerate(row_nodes):
                if i == 0:
                    node.left = row_nodes[-1]
                    row_nodes[-1].right = node
                else:
                    node.left = row_nodes[i-1]
                    row_nodes[i-1].right = node
            nodes_by_row[r] = row_nodes
    
    return root, headers, nodes_by_row

# %%
def cover_column(header):
    header.right.left = header.left
    header.left.right = header.right
    
    node = header.down
    while node != header:
        right_node = node.right
        while right_node != node:
            right_node.up.down = right_node.down
            right_node.down.up = right_node.up
            right_node.header.size -= 1
            right_node = right_node.right
        node = node.down

# %%
def uncover_column(header):
    node = header.up
    while node != header:
        left_node = node.left
        while left_node != node:
            left_node.up.down = left_node
            left_node.down.up = left_node
            left_node.header.size += 1
            left_node = left_node.left
        node = node.up
    
    header.right.left = header
    header.left.right = header

# %%
def dlx_with_gif(n, givens, output_gif='dlx_algorithm.gif'):
    import tempfile
    import imageio
    import os
    
    M, N = build_exact_cover(n)
    M_dense = M.toarray()
    
    given_rows = set()
    for x, y, value in givens:
        given_row = x * N * N + y * N + value
        given_rows.add(given_row)
    
    covered_cols = set()
    for given_row in given_rows:
        cols = np.where(M_dense[given_row, :] == 1)[0]
        covered_cols.update(cols)
    
    conflicting_rows = set()
    for col in covered_cols:
        rows_with_one = np.where(M_dense[:, col] == 1)[0]
        conflicting_rows.update(rows_with_one)
    conflicting_rows -= given_rows
    
    root, headers, nodes_by_row = build_dlx_structure(M, N)
    
    for given_row in given_rows:
        if given_row in nodes_by_row:
            for node in nodes_by_row[given_row]:
                if node.header.size > 0:
                    cover_column(node.header)
    
    for conflicting_row in conflicting_rows:
        if conflicting_row in nodes_by_row:
            for node in nodes_by_row[conflicting_row]:
                if node.header.size > 0:
                    cover_column(node.header)
    
    temp_dir = tempfile.mkdtemp()
    image_files = []
    frame_count = 0
    
    solution = []
    for x, y, value in givens:
        given_row = x * N * N + y * N + value
        solution.append(given_row)
    
    def solve(root, sol, depth=0):
        nonlocal frame_count
        
        if root.right == root:
            final_choices = []
            for row in sol:
                x = row // (N * N)
                y = (row // N) % N
                value = row % N
                final_choices.append((x, y, value))
            save_plot_with_choice(M, N, final_choices, os.path.join(temp_dir, f'frame_{frame_count:04d}.png'))
            image_files.append(os.path.join(temp_dir, f'frame_{frame_count:04d}.png'))
            frame_count += 1
            return sol
        
        col = root.right
        c = col
        while c != root:
            if c.size < col.size:
                col = c
            c = c.right
        
        if col.size == 0:
            return None
        
        cover_column(col)
        
        node = col.down
        while node != col:
            new_sol = sol + [node.row]
            
            current_choices = []
            for r in new_sol:
                x = r // (N * N)
                y = (r // N) % N
                v = r % N
                current_choices.append((x, y, v))
            
            save_plot_with_choice(M, N, current_choices, os.path.join(temp_dir, f'frame_{frame_count:04d}.png'))
            image_files.append(os.path.join(temp_dir, f'frame_{frame_count:04d}.png'))
            frame_count += 1
            
            right_node = node.right
            while right_node != node:
                cover_column(right_node.header)
                right_node = right_node.right
            
            result = solve(root, new_sol, depth + 1)
            if result is not None:
                return result
            
            left_node = node.left
            while left_node != node:
                uncover_column(left_node.header)
                left_node = left_node.left
            
            node = node.down
        
        uncover_column(col)
        return None
    
    result = solve(root, solution)
    
    if result is None:
        print("No solution found")
        return None
    
    final_choices = []
    for row in result:
        x = row // (N * N)
        y = (row // N) % N
        value = row % N
        final_choices.append((x, y, value))
    
    if len(image_files) > 0:
        from PIL import Image
        images = []
        target_size = None
        for f in image_files:
            img = Image.open(f)
            if target_size is None:
                target_size = img.size
            else:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(np.array(img))
        imageio.mimsave(output_gif, images, duration=0.1)
        print(f"GIF saved to {output_gif}")
    
    for f in image_files:
        os.remove(f)
    os.rmdir(temp_dir)
    
    return final_choices

# %%
M, N = build_exact_cover(n=2)
print(f"N = {N}")
print(f"Matrix shape: {M.shape}")
print(f"Number of ones (nnz): {M.nnz}")

plot_sparsity(M, N)

# %%

choices = [(0, 1, 0)]

plot_with_choice(M, N, choices)

# %%

plot_sudoku(N, choices)

# %%

final_choices = algorithm_x_with_gif(n=3, givens=choices)

plot_sudoku(N, final_choices)

# %%

# %%

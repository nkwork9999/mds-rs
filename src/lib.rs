//! Pure Rust MDS (Multidimensional Scaling) Library
//! Similar to sklearn.manifold.MDS

use std::f64;

// ============================================================================
// 設定・構造体
// ============================================================================

/// MDS設定
#[derive(Clone, Debug)]
pub struct MdsConfig {
    /// 出力次元数
    pub n_components: usize,
    /// 最大イテレーション数
    pub max_iter: usize,
    /// 収束判定のための許容誤差
    pub eps: f64,
    /// メトリックMDSかノンメトリックMDSか
    pub metric: bool,
    /// 乱数シード
    pub random_seed: u64,
    /// 初期配置
    pub init: Option<Vec<Vec<f64>>>,
}

impl Default for MdsConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 300,
            eps: 1e-6,
            metric: true,
            random_seed: 42,
            init: None,
        }
    }
}

impl MdsConfig {
    /// 出力次元数を指定して作成
    pub fn with_components(n_components: usize) -> Self {
        Self {
            n_components,
            ..Default::default()
        }
    }

    /// 非計量MDSの設定を作成
    pub fn nonmetric() -> Self {
        Self {
            metric: false,
            ..Default::default()
        }
    }
}

/// MDS結果
#[derive(Debug, Clone)]
pub struct MdsResult {
    /// 埋め込み座標
    pub embedding: Vec<Vec<f64>>,
    /// 最終ストレス値
    pub stress: f64,
    /// イテレーション回数
    pub n_iter: usize,
}

/// 疑似乱数生成器
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ============================================================================
// 基本関数
// ============================================================================

/// ユークリッド距離行列を計算
pub fn euclidean_distance_matrix(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut dist = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i + 1..n {
            let d = euclidean_distance(&points[i], &points[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

/// 2点間のユークリッド距離
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// ストレス値を計算（Kruskal's stress-1）
pub fn compute_stress(embedding: &[Vec<f64>], distances: &[Vec<f64>]) -> f64 {
    let n = embedding.len();
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        for j in i + 1..n {
            let d_ij = distances[i][j];
            let emb_dist = euclidean_distance(&embedding[i], &embedding[j]);
            numerator += (d_ij - emb_dist).powi(2);
            denominator += d_ij.powi(2);
        }
    }

    if denominator > 0.0 {
        (numerator / denominator).sqrt()
    } else {
        0.0
    }
}

fn compute_raw_stress(embedding: &[Vec<f64>], distances: &[Vec<f64>]) -> f64 {
    let n = embedding.len();
    let mut stress = 0.0;

    for i in 0..n {
        for j in i + 1..n {
            let d_ij = distances[i][j];
            let emb_dist = euclidean_distance(&embedding[i], &embedding[j]);
            stress += (d_ij - emb_dist).powi(2);
        }
    }
    stress
}

/// Classical MDS（主座標分析）
pub fn classical_mds(distances: &[Vec<f64>], n_components: usize) -> MdsResult {
    let n = distances.len();

    if n == 0 {
        return MdsResult { embedding: Vec::new(), stress: 0.0, n_iter: 0 };
    }

    let mut d_sq = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            d_sq[i][j] = distances[i][j].powi(2);
        }
    }

    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    let mut grand_mean = 0.0;

    for i in 0..n {
        for j in 0..n {
            row_means[i] += d_sq[i][j];
            col_means[j] += d_sq[i][j];
            grand_mean += d_sq[i][j];
        }
    }

    for i in 0..n {
        row_means[i] /= n as f64;
        col_means[i] /= n as f64;
    }
    grand_mean /= (n * n) as f64;

    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            b[i][j] = -0.5 * (d_sq[i][j] - row_means[i] - col_means[j] + grand_mean);
        }
    }

    let eigenvectors = power_iteration_multiple(&b, n_components);
    let stress = compute_stress(&eigenvectors, distances);

    MdsResult { embedding: eigenvectors, stress, n_iter: 0 }
}

fn power_iteration_multiple(matrix: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let k = k.min(n);
    let max_iter = 1000;
    let tol = 1e-10;

    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut deflated = matrix.to_vec();

    for _ in 0..k {
        let mut rng = SimpleRng::new(42);
        let mut v: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        normalize(&mut v);

        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += deflated[i][j] * v[j];
                }
            }

            let new_eigenvalue: f64 = new_v.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            normalize(&mut new_v);

            let diff: f64 = v.iter().zip(new_v.iter()).map(|(a, b)| (a - b).abs()).sum();
            v = new_v;
            eigenvalue = new_eigenvalue;

            if diff < tol {
                break;
            }
        }

        if eigenvalue > 0.0 {
            let scaled: Vec<f64> = v.iter().map(|x| x * eigenvalue.sqrt()).collect();
            eigenvectors.push(scaled);

            for i in 0..n {
                for j in 0..n {
                    deflated[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
    }

    let mut result = vec![vec![0.0; eigenvectors.len()]; n];
    for (comp, evec) in eigenvectors.iter().enumerate() {
        for i in 0..n {
            result[i][comp] = evec[i];
        }
    }
    result
}

fn normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// SMACOF アルゴリズム
pub fn smacof(distances: &[Vec<f64>], config: &MdsConfig) -> MdsResult {
    let n = distances.len();

    if n == 0 {
        return MdsResult { embedding: Vec::new(), stress: 0.0, n_iter: 0 };
    }

    let k = config.n_components;

    let mut x = if let Some(ref init) = config.init {
        init.clone()
    } else {
        let mut rng = SimpleRng::new(config.random_seed);
        (0..n).map(|_| (0..k).map(|_| rng.next_normal()).collect()).collect()
    };

    let w = vec![vec![1.0; n]; n];
    let v_plus_factor = 1.0 / n as f64;

    let mut old_stress = compute_raw_stress(&x, distances);
    let mut n_iter = 0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let d_x = euclidean_distance_matrix(&x);

        let mut b_x = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j && d_x[i][j] > 1e-10 {
                    b_x[i][j] = -w[i][j] * distances[i][j] / d_x[i][j];
                }
            }
            b_x[i][i] = -b_x[i].iter().sum::<f64>() + b_x[i][i];
        }

        let mut new_x = vec![vec![0.0; k]; n];
        for i in 0..n {
            for dim in 0..k {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += b_x[i][j] * x[j][dim];
                }
                new_x[i][dim] = sum * v_plus_factor;
            }
        }

        x = new_x;
        center_embedding(&mut x);

        let new_stress = compute_raw_stress(&x, distances);

        if (old_stress - new_stress).abs() < config.eps {
            break;
        }
        old_stress = new_stress;
    }

    let final_stress = compute_stress(&x, distances);
    MdsResult { embedding: x, stress: final_stress, n_iter }
}

fn center_embedding(x: &mut [Vec<f64>]) {
    if x.is_empty() {
        return;
    }

    let n = x.len();
    let k = x[0].len();

    for dim in 0..k {
        let mean: f64 = x.iter().map(|row| row[dim]).sum::<f64>() / n as f64;
        for row in x.iter_mut() {
            row[dim] -= mean;
        }
    }
}

/// MDS（Classical + SMACOF）
pub fn mds(distances: &[Vec<f64>], config: &MdsConfig) -> MdsResult {
    let classical = classical_mds(distances, config.n_components);
    let smacof_config = MdsConfig {
        init: Some(classical.embedding),
        ..config.clone()
    };
    smacof(distances, &smacof_config)
}

/// Non-metric MDS
pub fn nonmetric_mds(distances: &[Vec<f64>], config: &MdsConfig) -> MdsResult {
    let n = distances.len();

    if n == 0 {
        return MdsResult { embedding: Vec::new(), stress: 0.0, n_iter: 0 };
    }

    let metric_result = mds(distances, config);
    let mut x = metric_result.embedding;
    let k = config.n_components;

    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            pairs.push((i, j, distances[i][j]));
        }
    }
    pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let mut n_iter = 0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let d_x = euclidean_distance_matrix(&x);
        let disparities = isotonic_regression(&pairs, &d_x);

        let mut disparity_matrix = vec![vec![0.0; n]; n];
        for (idx, &(i, j, _)) in pairs.iter().enumerate() {
            disparity_matrix[i][j] = disparities[idx];
            disparity_matrix[j][i] = disparities[idx];
        }

        let w = vec![vec![1.0; n]; n];
        let v_plus_factor = 1.0 / n as f64;

        let mut b_x = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j && d_x[i][j] > 1e-10 {
                    b_x[i][j] = -w[i][j] * disparity_matrix[i][j] / d_x[i][j];
                }
            }
            b_x[i][i] = -b_x[i].iter().sum::<f64>() + b_x[i][i];
        }

        let mut new_x = vec![vec![0.0; k]; n];
        for i in 0..n {
            for dim in 0..k {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += b_x[i][j] * x[j][dim];
                }
                new_x[i][dim] = sum * v_plus_factor;
            }
        }

        let diff: f64 = x.iter().zip(new_x.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>().sqrt();

        x = new_x;
        center_embedding(&mut x);

        if diff < config.eps {
            break;
        }
    }

    let final_stress = compute_stress(&x, distances);
    MdsResult { embedding: x, stress: final_stress, n_iter }
}

fn isotonic_regression(pairs: &[(usize, usize, f64)], d_x: &[Vec<f64>]) -> Vec<f64> {
    let n = pairs.len();
    let mut y: Vec<f64> = pairs.iter().map(|&(i, j, _)| d_x[i][j]).collect();

    let mut blocks: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();

    loop {
        let mut changed = false;
        let mut new_blocks = Vec::new();
        let mut i = 0;

        while i < blocks.len() {
            if i + 1 < blocks.len() {
                let (start1, end1) = blocks[i];
                let (start2, end2) = blocks[i + 1];

                let mean1: f64 = (start1..=end1).map(|j| y[j]).sum::<f64>() / (end1 - start1 + 1) as f64;
                let mean2: f64 = (start2..=end2).map(|j| y[j]).sum::<f64>() / (end2 - start2 + 1) as f64;

                if mean1 > mean2 {
                    let new_mean: f64 = (start1..=end2).map(|j| y[j]).sum::<f64>() / (end2 - start1 + 1) as f64;
                    for j in start1..=end2 {
                        y[j] = new_mean;
                    }
                    new_blocks.push((start1, end2));
                    i += 2;
                    changed = true;
                    continue;
                }
            }
            new_blocks.push(blocks[i]);
            i += 1;
        }

        blocks = new_blocks;
        if !changed {
            break;
        }
    }
    y
}

/// Procrustes分析
pub fn procrustes(x: &[Vec<f64>], y: &[Vec<f64>]) -> (Vec<Vec<f64>>, f64) {
    let n = x.len();
    if n == 0 || y.len() != n {
        return (Vec::new(), f64::INFINITY);
    }

    let k = x[0].len();
    if y[0].len() != k {
        return (Vec::new(), f64::INFINITY);
    }

    let mut x_centered = x.to_vec();
    let mut y_centered = y.to_vec();
    center_embedding(&mut x_centered);
    center_embedding(&mut y_centered);

    let scale_x: f64 = x_centered.iter().flat_map(|row| row.iter()).map(|v| v.powi(2)).sum::<f64>().sqrt();
    let scale_y: f64 = y_centered.iter().flat_map(|row| row.iter()).map(|v| v.powi(2)).sum::<f64>().sqrt();

    for row in x_centered.iter_mut() {
        for v in row.iter_mut() { *v /= scale_x; }
    }
    for row in y_centered.iter_mut() {
        for v in row.iter_mut() { *v /= scale_y; }
    }

    if k == 2 {
        let mut a = 0.0;
        let mut b = 0.0;
        for i in 0..n {
            a += y_centered[i][0] * x_centered[i][0] + y_centered[i][1] * x_centered[i][1];
            b += y_centered[i][0] * x_centered[i][1] - y_centered[i][1] * x_centered[i][0];
        }
        let norm = (a.powi(2) + b.powi(2)).sqrt();
        let cos_t = a / norm;
        let sin_t = b / norm;

        let mut y_transformed = vec![vec![0.0; k]; n];
        for i in 0..n {
            y_transformed[i][0] = cos_t * y_centered[i][0] - sin_t * y_centered[i][1];
            y_transformed[i][1] = sin_t * y_centered[i][0] + cos_t * y_centered[i][1];
        }

        let disparity: f64 = x_centered.iter().zip(y_transformed.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .map(|(a, b)| (a - b).powi(2)).sum();

        (y_transformed, disparity)
    } else {
        let disparity: f64 = x_centered.iter().zip(y_centered.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .map(|(a, b)| (a - b).powi(2)).sum();

        (y_centered, disparity)
    }
}

/// ランダムな距離行列を生成
pub fn random_distance_matrix(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = SimpleRng::new(seed);
    let mut dist = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i + 1..n {
            let d = rng.next_f64() * 10.0;
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_matrix() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let dist = euclidean_distance_matrix(&points);
        assert!((dist[0][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_classical_mds() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866]];
        let dist = euclidean_distance_matrix(&points);
        let result = classical_mds(&dist, 2);
        assert!(result.stress < 0.1);
    }

    #[test]
    fn test_stress_zero() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let dist = euclidean_distance_matrix(&points);
        let stress = compute_stress(&points, &dist);
        assert!((stress - 0.0).abs() < 1e-10);
    }
}
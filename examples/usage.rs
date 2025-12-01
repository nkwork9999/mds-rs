//! MDS (Multidimensional Scaling) ライブラリ使用例
//!
//! 実行方法:
//!   cargo run --example usage

use mds_rs::{
    mds, classical_mds, smacof, nonmetric_mds,
    compute_stress, procrustes, euclidean_distance_matrix,
    random_distance_matrix, MdsConfig,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           MDS ライブラリ使用例                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    example_basic();
    example_dimension_reduction();
    example_algorithms();
    example_procrustes();
    example_stress();
    example_random();

    println!("\n✓ すべての使用例が完了しました");
}

/// 1. 基本的なMDS（距離行列からの座標復元）
fn example_basic() {
    println!("【1】基本的なMDS（距離→座標復元）");
    println!("─────────────────────────────────────────");
    
    // 元の2D座標
    let original_points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    
    // 距離行列を計算
    let distances = euclidean_distance_matrix(&original_points);
    
    // MDSで座標を復元
    let config = MdsConfig::with_components(2);
    let result = mds(&distances, &config);
    
    println!("  元の座標:");
    for (i, p) in original_points.iter().enumerate() {
        println!("    Point {}: ({:.1}, {:.1})", i, p[0], p[1]);
    }
    
    println!("\n  MDSで復元した座標:");
    for (i, p) in result.embedding.iter().enumerate() {
        println!("    Point {}: ({:.4}, {:.4})", i, p[0], p[1]);
    }
    
    println!("\n  Stress値: {:.6} (0に近いほど良い)", result.stress);
    println!("  反復回数: {}\n", result.n_iter);
}

/// 2. 次元削減
fn example_dimension_reduction() {
    println!("【2】次元削減 (3D → 2D → 1D)");
    println!("─────────────────────────────────────────");
    
    // 3D座標
    let points_3d = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    
    let distances = euclidean_distance_matrix(&points_3d);
    
    // 2Dに削減
    let result_2d = mds(&distances, &MdsConfig::with_components(2));
    // 1Dに削減
    let result_1d = mds(&distances, &MdsConfig::with_components(1));
    
    println!("  3D入力:");
    for (i, p) in points_3d.iter().enumerate() {
        println!("    Point {}: ({:.1}, {:.1}, {:.1})", i, p[0], p[1], p[2]);
    }
    
    println!("\n  2Dへの削減 (Stress: {:.4}):", result_2d.stress);
    for (i, p) in result_2d.embedding.iter().enumerate() {
        println!("    Point {}: ({:.4}, {:.4})", i, p[0], p[1]);
    }
    
    println!("\n  1Dへの削減 (Stress: {:.4}):", result_1d.stress);
    for (i, p) in result_1d.embedding.iter().enumerate() {
        println!("    Point {}: ({:.4})", i, p[0]);
    }
    
    println!("\n  ※ 次元を減らすほどStress値が大きくなる\n");
}

/// 3. 各アルゴリズムの比較
fn example_algorithms() {
    println!("【3】MDSアルゴリズム比較");
    println!("─────────────────────────────────────────");
    
    // テスト用の距離行列
    let points = vec![
        vec![0.0, 0.0], vec![1.0, 0.0],
        vec![0.5, 0.866], vec![1.5, 0.866],
    ];
    let distances = euclidean_distance_matrix(&points);
    
    let config = MdsConfig::default();
    
    // Classical MDS
    let classical = classical_mds(&distances, 2);
    
    // SMACOF
    let smacof_result = smacof(&distances, &config);
    
    // Combined (Classical + SMACOF)
    let combined = mds(&distances, &config);
    
    // Non-metric MDS
    let nonmetric = nonmetric_mds(&distances, &MdsConfig::nonmetric());
    
    println!("  アルゴリズム        | Stress値   | 反復回数");
    println!("  ─────────────────────────────────────────");
    println!("  Classical MDS       | {:.6}   | -", classical.stress);
    println!("  SMACOF              | {:.6}   | {}", smacof_result.stress, smacof_result.n_iter);
    println!("  Combined            | {:.6}   | {}", combined.stress, combined.n_iter);
    println!("  Non-metric MDS      | {:.6}   | {}", nonmetric.stress, nonmetric.n_iter);
    println!();
}

/// 4. Procrustes分析
fn example_procrustes() {
    println!("【4】Procrustes分析（形状比較）");
    println!("─────────────────────────────────────────");
    
    // 2つの三角形
    let shape1 = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.5, 0.866],
    ];
    
    // 回転・スケール変換された三角形
    let shape2 = vec![
        vec![0.0, 0.0],
        vec![0.0, 2.0],
        vec![1.732, 1.0],
    ];
    
    let (transformed, disparity) = procrustes(&shape1, &shape2);
    
    println!("  形状1 (基準):");
    for (i, p) in shape1.iter().enumerate() {
        println!("    Point {}: ({:.3}, {:.3})", i, p[0], p[1]);
    }
    
    println!("\n  形状2 (比較対象):");
    for (i, p) in shape2.iter().enumerate() {
        println!("    Point {}: ({:.3}, {:.3})", i, p[0], p[1]);
    }
    
    println!("\n  変換後の形状2:");
    for (i, p) in transformed.iter().enumerate() {
        println!("    Point {}: ({:.4}, {:.4})", i, p[0], p[1]);
    }
    
    println!("\n  Procrustes距離: {:.6}", disparity);
    println!("  ※ 小さいほど形状が似ている\n");
}

/// 5. ストレス値の解釈
fn example_stress() {
    println!("【5】ストレス値の解釈");
    println!("─────────────────────────────────────────");
    
    // 完璧な埋め込み
    let points = vec![
        vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0],
        vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 1.0],
    ];
    let distances = euclidean_distance_matrix(&points);
    let perfect_stress = compute_stress(&points, &distances);
    
    // 悪い埋め込み（スケールが違う）
    let bad_points = vec![
        vec![0.0, 0.0], vec![0.5, 0.0], vec![1.0, 0.0],
        vec![0.0, 0.5], vec![0.5, 0.5], vec![1.0, 0.5],
    ];
    let bad_stress = compute_stress(&bad_points, &distances);
    
    println!("  完璧な埋め込み: Stress = {:.6}", perfect_stress);
    println!("  悪い埋め込み:   Stress = {:.6}", bad_stress);
    println!();
    println!("  Stress値の目安:");
    println!("    0.00 - 0.05: 優秀（完璧な埋め込み）");
    println!("    0.05 - 0.10: 良好（ほぼ正確）");
    println!("    0.10 - 0.20: まあまあ（概形は保持）");
    println!("    0.20以上:    要改善（情報損失大）\n");
}

/// 6. ランダム距離行列からのMDS
fn example_random() {
    println!("【6】ランダム距離行列からのMDS");
    println!("─────────────────────────────────────────");
    
    let n = 8;
    let random_dist = random_distance_matrix(n, 42);
    
    // 異なる次元数での結果
    let result_1d = mds(&random_dist, &MdsConfig::with_components(1));
    let result_2d = mds(&random_dist, &MdsConfig::with_components(2));
    let result_3d = mds(&random_dist, &MdsConfig::with_components(3));
    
    println!("  {}点のランダム距離行列を各次元に埋め込み:", n);
    println!();
    
    let eval = |s: f64| -> &str {
        if s < 0.05 { "優秀" }
        else if s < 0.10 { "良好" }
        else if s < 0.20 { "まあまあ" }
        else { "要改善" }
    };
    
    println!("  次元 | Stress値   | 評価");
    println!("  ─────────────────────────────────────────");
    println!("   1D  | {:.6}   | {}", result_1d.stress, eval(result_1d.stress));
    println!("   2D  | {:.6}   | {}", result_2d.stress, eval(result_2d.stress));
    println!("   3D  | {:.6}   | {}", result_3d.stress, eval(result_3d.stress));
    
    println!("\n  2D埋め込み結果:");
    for (i, p) in result_2d.embedding.iter().enumerate() {
        println!("    Point {}: ({:7.4}, {:7.4})", i, p[0], p[1]);
    }
}
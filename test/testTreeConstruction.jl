tree = PFTDPWTree{Float64, Float64}()
b0 = WeightedParticleBelief([1,2],[0.5,0.5])
a0 = 1.0

b1 = WeightedParticleBelief([5,4],[0.25,0.75])
o = 5.0

@testset "Initial Belief Insertion" begin

    insert_belief!(tree, b0, 0, 0.0)
    @test tree.Nh[1] == 0
    @test length(tree.b) == 1
    @test tree.b[1] == b0
    @test length(tree.b_children) == 1
    @test length(tree.b_parent) == 1 # Parent of initial belief is node 0
    @test tree.n_b == length(tree.b)
    @test tree.b_parent[1] == 0
end

@testset "Belief-action Node Insertion" begin
    insert_action!(tree, 1, a0)

    # Ensure nothing is inserted in belief node spots
    @test tree.n_b == length(tree.b)
    @test tree.Nh[1] == 0
    @test length(tree.b) == 1
    @test tree.b[1] == b0
    @test length(tree.b_children) == 1
    @test length(tree.b_parent) == 1

    # Test ba insertion
    @test tree.Nha[1] == 0
    @test tree.Qha[1] == 0.0
    @test tree.b_children[1][a0] == 1 # First action leads to first ba node
    @test tree.ba_parent[1] == 1
    @test tree.n_ba == 1
end

# insert new belief node
@testset "New Belief Node" begin
    ba_idx = 1
    insert_belief!(tree, b1, ba_idx, o)

    @test tree.n_b == length(tree.b)
    @test length(tree.b) == 2
    @test tree.b[2] == b1
    @test length(tree.b_children) == 2
    @test length(tree.b_parent) == 2
    @test tree.b_parent[2] == 1
end

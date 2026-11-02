# Copies what the scripts write under datasets/ to a mirror of that directory elsewhere (a Dropbox folder, say), when
# the checkout names one: the environment variable SLOWLIGHT_MIRROR, or else the first line of the untracked file
# `.datasets_mirror` at the repository's root. Output outside datasets/ (a quick look in scratch/, say) stays put.
module Mirror

export mirror_dataset

const DATASETS = normpath(joinpath(@__DIR__, "..", "datasets"))
const MIRROR_FILE = normpath(joinpath(@__DIR__, "..", ".datasets_mirror"))

"The mirror's root directory, or `nothing` when none is set."
function mirror_root()
    root = get(ENV, "SLOWLIGHT_MIRROR", "")
    if isempty(root) && isfile(MIRROR_FILE)
        lines = filter(!isempty, strip.(readlines(MIRROR_FILE)))
        root = isempty(lines) ? "" : first(lines)
    end
    return isempty(root) ? nothing : normpath(expanduser(root))
end

"""
    mirror_dataset(path)

Copy `path`, a dataset directory or file under datasets/, to the same place under the mirror, file by file: files
there are replaced, and files only in the mirror are kept. Does nothing when no mirror is set or `path` lies outside
datasets/; a failure is a warning, not an error, since the dataset itself is already written.
"""
function mirror_dataset(path)
    try
        root = mirror_root(); root === nothing && return nothing
        src = normpath(abspath(path)); rel = relpath(src, DATASETS)
        (rel == "." || startswith(rel, "..")) && return nothing
        isdir(root) || (@warn "the mirror directory does not exist; dataset not copied" root; return nothing)
        files = isdir(src) ? [joinpath(d, f) for (d, _, fs) in walkdir(src) for f in fs] : [src]
        for f in files
            dst = joinpath(root, relpath(f, DATASETS)); mkpath(dirname(dst))
            cp(f, dst; force = true)
        end
        @info "mirrored" to = joinpath(root, rel) files = length(files)
    catch err
        @warn "dataset not mirrored" path exception = err
    end
    flush(stderr)
    return nothing
end

end

function out = cmp_join(parts)
    sep = "/";
    out = parts(1);
    for i = 2:size(parts, "*")
        if part(out, length(out)) == sep then
            out = out + parts(i);
        else
            out = out + sep + parts(i);
        end
    end
endfunction

function cmp_ensure_dir(path)
    if ~isdir(path) then
        mkdir(path);
    end
endfunction

function data = cmp_read_tsv(path)
    if ~isfile(path) then
        data = [];
        return
    end
    data = csvRead(path, ascii(9));
endfunction

function value = cmp_read_scalar(path, default_value)
    data = cmp_read_tsv(path);
    if size(data, "*") == 0 then
        value = default_value;
    else
        value = data(1, 1);
    end
endfunction

function cmp_write_tsv(path, data)
    if size(data, "*") == 0 then
        mputl("", path);
        return
    end

    matrix_data = matrix(data, size(data, 1), size(data, 2));
    lines = strings(size(matrix_data, 1), 1);
    for i = 1:size(matrix_data, 1)
        row_text = "";
        for j = 1:size(matrix_data, 2)
            cell_text = msprintf("%.18e", matrix_data(i, j));
            if j == 1 then
                row_text = cell_text;
            else
                row_text = row_text + ascii(9) + cell_text;
            end
        end
        lines(i) = row_text;
    end
    mputl(lines, path);
endfunction

function cmp_write_if_present(outdir, name, data)
    if size(data, "*") <> 0 then
        cmp_write_tsv(cmp_join([outdir, name + ".tsv"]), data);
    end
endfunction

function R = cmp_reaction(q, C, dof)
    if size(C, "*") == 0 then
        R = [];
        return
    end

    if dof == 1 then
        nodes = C(:, 1);
        reactions = q(nodes);
        R = [nodes, ones(size(nodes, 1), 1), reactions];
    else
        dof_no = (C(:, 1) - 1) * dof + C(:, 2);
        reactions = q(dof_no);
        R = [C(:, 1:2), reactions];
    end
endfunction

function cmp_export_common(outdir, X, T, G, C, P, u, q, S, E, R, f, U, F)
    cmp_ensure_dir(outdir);
    cmp_write_if_present(outdir, "X", X);
    cmp_write_if_present(outdir, "T", T);
    cmp_write_if_present(outdir, "G", G);
    cmp_write_if_present(outdir, "C", C);
    cmp_write_if_present(outdir, "P", P);
    cmp_write_if_present(outdir, "u", u);
    cmp_write_if_present(outdir, "q", q);
    cmp_write_if_present(outdir, "S", S);
    cmp_write_if_present(outdir, "E", E);
    cmp_write_if_present(outdir, "R", R);
    cmp_write_if_present(outdir, "f", f);
    cmp_write_if_present(outdir, "U", U);
    cmp_write_if_present(outdir, "F", F);
endfunction

function [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_elastic_q4(indir)
    X = cmp_read_tsv(cmp_join([indir, "X.tsv"]));
    T = cmp_read_tsv(cmp_join([indir, "T.tsv"]));
    G = cmp_read_tsv(cmp_join([indir, "G.tsv"]));
    C = cmp_read_tsv(cmp_join([indir, "C.tsv"]));
    P = cmp_read_tsv(cmp_join([indir, "P.tsv"]));
    dof = cmp_read_scalar(cmp_join([indir, "dof.tsv"]), 2);

    [K, p, q] = init(size(X, 1), dof);
    K = kq4e(K, T, X, G);
    if size(P, "*") <> 0 then
        p = setload(p, P);
    end
    [K, p] = setbc(K, p, C, dof);
    u = K \ p;
    [q, S, E] = qq4e(q, T, X, G, u);
    R = cmp_reaction(q, C, dof);
    f = [];
    U = [];
    F = [];
endfunction

function [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_triangle_t3(indir)
    X = cmp_read_tsv(cmp_join([indir, "X.tsv"]));
    T = cmp_read_tsv(cmp_join([indir, "T.tsv"]));
    G = cmp_read_tsv(cmp_join([indir, "G.tsv"]));
    C = cmp_read_tsv(cmp_join([indir, "C.tsv"]));
    P = cmp_read_tsv(cmp_join([indir, "P.tsv"]));
    dof = cmp_read_scalar(cmp_join([indir, "dof.tsv"]), 2);

    [K, p, q] = init(size(X, 1), dof);
    K = kt3e(K, T, X, G);
    if size(P, "*") <> 0 then
        p = setload(p, P);
    end
    u = solve_lag(K, p, C, dof);
    [q, S, E] = qt3e(q, T, X, G, u);
    R = cmp_reaction(q, C, dof);
    f = [];
    U = [];
    F = [];
endfunction

function [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_flow(indir, use_quads)
    X = cmp_read_tsv(cmp_join([indir, "X.tsv"]));
    T = cmp_read_tsv(cmp_join([indir, "T.tsv"]));
    G = cmp_read_tsv(cmp_join([indir, "G.tsv"]));
    C = cmp_read_tsv(cmp_join([indir, "C.tsv"]));
    P = cmp_read_tsv(cmp_join([indir, "P.tsv"]));

    [K, p, q] = init(size(X, 1), 1);
    if use_quads then
        K = kq4p(K, T, X, G);
    else
        K = kt3p(K, T, X, G);
    end
    if size(P, "*") <> 0 then
        p = setload(p, P);
    end
    [K, p] = setbc(K, p, C, 1);
    u = K \ p;
    if use_quads then
        [q, S, E] = qq4p(q, T, X, G, u);
    else
        [q, S, E] = qt3p(q, T, X, G, u);
    end
    R = cmp_reaction(q, C, 1);
    f = [];
    U = [];
    F = [];
endfunction

function [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_nlbar(indir)
    X = cmp_read_tsv(cmp_join([indir, "X.tsv"]));
    T = cmp_read_tsv(cmp_join([indir, "T.tsv"]));
    G = cmp_read_tsv(cmp_join([indir, "G.tsv"]));
    C = cmp_read_tsv(cmp_join([indir, "C.tsv"]));
    P = cmp_read_tsv(cmp_join([indir, "P.tsv"]));
    no_loadsteps = cmp_read_scalar(cmp_join([indir, "no_loadsteps.tsv"]), 1);
    i_max = cmp_read_scalar(cmp_join([indir, "i_max.tsv"]), 8);
    i_d = cmp_read_scalar(cmp_join([indir, "i_d.tsv"]), 3);
    TOL = cmp_read_scalar(cmp_join([indir, "TOL.tsv"]), 1.0e-3);

    dof = size(X, 2);
    ndof = dof * size(X, 1);
    u = zeros(ndof, 1);
    du = zeros(ndof, 1);
    f = zeros(ndof, 1);
    df = zeros(ndof, 1);
    if size(P, "*") <> 0 then
        df = setload(df, P);
    end

    U = zeros(no_loadsteps + 1, 1);
    F = zeros(no_loadsteps + 1, 1);
    n = 1;
    i = i_d;
    K = [];
    du0 = [];

    while n <= no_loadsteps
        if i < i_max then
            K = zeros(ndof, ndof);
            K = kbar(K, T, X, G, u);
            [Kt, df] = setbc(K, df, C, dof);
            du0 = Kt \ df;

            if du' * du0 < 0 then
                df = -df;
                du0 = -du0;
            end

            if n == 1 then
                l0 = norm(du0);
                l = l0;
                l_max = 2 * l0;
            else
                l = norm(du);
                l0 = norm(du0);
            end
        end

        if i_d <= i & i < i_max then
            du = min(l / l0, l_max / l0) * du0;
        elseif i < i_d then
            du = min(2 * l / l0, l_max / l0) * du0;
        else
            du0 = 0.5 * du0;
            du = du0;
        end

        for i = 1:i_max
            q = zeros(ndof, 1);
            [q, S, E] = qbar(q, T, X, G, u + du);
            dq = q - f;
            xi = (dq' * du) / (df' * du);
            r = -dq + xi * df;

            if rnorm(r, C, dof) < TOL * rnorm(df, C, dof) then
                break
            else
                [Kt, r] = setbc(K, r, C, dof);
                delta_u = Kt \ r;
                du = du + delta_u;
            end
        end

        if i < i_max then
            f = f + xi * df;
            u = u + du;
            U(n + 1) = u($);
            F(n + 1) = f($);
            n = n + 1;
        end
    end

    q = zeros(ndof, 1);
    [q, S, E] = qbar(q, T, X, G, u);
    R = cmp_reaction(q, C, dof);
endfunction

function [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_plastic(indir, plane_strain)
    X = cmp_read_tsv(cmp_join([indir, "X.tsv"]));
    T = cmp_read_tsv(cmp_join([indir, "T.tsv"]));
    G = cmp_read_tsv(cmp_join([indir, "G.tsv"]));
    C = cmp_read_tsv(cmp_join([indir, "C.tsv"]));
    P = cmp_read_tsv(cmp_join([indir, "P.tsv"]));
    no_loadsteps = cmp_read_scalar(cmp_join([indir, "no_loadsteps.tsv"]), 1);
    i_max = cmp_read_scalar(cmp_join([indir, "i_max.tsv"]), 20);
    i_d = cmp_read_scalar(cmp_join([indir, "i_d.tsv"]), 8);
    TOL = cmp_read_scalar(cmp_join([indir, "TOL.tsv"]), 1.0e-2);
    mattype = 1;
    dof = 2;
    ndof = dof * size(X, 1);
    nelem = size(T, 1);

    f = zeros(ndof, 1);
    df = zeros(ndof, 1);
    if size(P, "*") <> 0 then
        df = setload(df, P);
    end
    u = zeros(ndof, 1);
    du = zeros(ndof, 1);
    S = zeros(nelem, 1);
    E = zeros(nelem, 1);
    U = zeros(no_loadsteps + 1, 1);
    F = zeros(no_loadsteps + 1, 1);
    n = 1;
    i = i_d;
    K = [];
    du0 = [];

    while n <= no_loadsteps
        if i < i_max then
            K = zeros(ndof, ndof);
            if plane_strain then
                K = kq4epe(K, T, X, G, S, E, mattype);
            else
                K = kq4eps(K, T, X, G, S, E, mattype);
            end
            [Kt, df] = setbc(K, df, C, dof);
            du0 = Kt \ df;

            if du' * du0 < 0 then
                df = -df;
                du0 = -du0;
            end

            if n == 1 then
                l0 = norm(du0);
                l = l0;
                l_max = 2 * l0;
            else
                l = norm(du);
                l0 = norm(du0);
            end
        end

        if i_d <= i & i < i_max then
            du = min(l / l0, l_max / l0) * du0;
        elseif i < i_d then
            du = min(2 * l / l0, l_max / l0) * du0;
        else
            du0 = 0.5 * du0;
            du = du0;
        end

        for i = 1:i_max
            q = zeros(ndof, 1);
            if plane_strain then
                [q, Sn, En] = qq4epe(q, T, X, G, u + du, S, E, mattype);
            else
                [q, Sn, En] = qq4eps(q, T, X, G, u + du, S, E, mattype);
            end

            dq = q - f;
            xi = (dq' * du) / (df' * du);
            r = -dq + xi * df;

            if rnorm(r, C, dof) < TOL * rnorm(df, C, dof) then
                break
            else
                [Kt, r] = setbc(K, r, C, dof);
                delta_u = Kt \ r;
                du = du + delta_u;
            end
        end

        if i < i_max then
            f = f + xi * df;
            u = u + du;
            S = Sn;
            E = En;
            U(n + 1) = u($);
            F(n + 1) = f($);
            n = n + 1;
        end
    end

    q = zeros(ndof, 1);
    if plane_strain then
        [q, S, E] = qq4epe(q, T, X, G, u, S, E, mattype);
    else
        [q, S, E] = qq4eps(q, T, X, G, u, S, E, mattype);
    end
    R = cmp_reaction(q, C, dof);
endfunction

function [Sd, Sm] = devstress(S)
    [Sd, Sm] = devstres(S);
endfunction

function cmp_main()
    mode(0);
    ieee(1);

    repo = pwd();
    case_name = getenv("FEMLAB_CASE");
    if case_name == "" then
        error("FEMLAB_CASE environment variable is required.");
    end

    disp("cmp:start:" + case_name);
    getd(cmp_join([repo, "macros"]));

    root = cmp_join([repo, "_solver_comparasion"]);
    indir = cmp_join([root, "inputs", case_name]);
    outdir = cmp_join([root, "raw", "scilab", case_name]);

    select case_name
    case "cantilever_q4" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_elastic_q4(indir);
    case "gmsh_triangle_t3" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_triangle_t3(indir);
    case "flow_q4" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_flow(indir, %t);
    case "flow_t3" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_flow(indir, %f);
    case "bar01_nlbar" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_nlbar(indir);
    case "bar02_nlbar" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_nlbar(indir);
    case "bar03_nlbar" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_nlbar(indir);
    case "square_plastps" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_plastic(indir, %f);
    case "square_plastpe" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_plastic(indir, %t);
    case "hole_plastps" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_plastic(indir, %f);
    case "hole_plastpe" then
        [X, T, G, C, P, u, q, S, E, R, f, U, F] = cmp_run_plastic(indir, %t);
    else
        error(msprintf("Unknown FEMLAB_CASE: %s", case_name));
    end

    disp("cmp:solved:" + case_name);
    cmp_export_common(outdir, X, T, G, C, P, u, q, S, E, R, f, U, F);
    disp("cmp:wrote:" + case_name);
endfunction

cmp_main();
exit;

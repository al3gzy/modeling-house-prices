function [beta_konacno, izabrane_kolone, z_score] = backwards_stepwise(X_ulaz, y_izlaz)

    X_trenutno = X_ulaz;
    [N, D] = size(X_trenutno);

    beta_full = (X_trenutno' * X_trenutno) \ (X_trenutno' * y_izlaz);
    rss_stari = rss(X_trenutno, y_izlaz, beta_full);
    sigma2 = rss_stari / (N - D);

    varijansa_beta = diag(inv(X_trenutno' * X_trenutno) * sigma2);
    z_score = beta_full ./ sqrt(varijansa_beta);

    F_score = 0;
    kriticna_vrednost = 1;
    izabrane_kolone = 1:D;
    izbacene_kolone = [];
    izbaceni_indeksi = [];

    while F_score < kriticna_vrednost && length(izabrane_kolone) > 1
        [~, indeks_najslabije] = min(abs(z_score));

        poslednje_izbacena = izabrane_kolone(indeks_najslabije);
        izbacene_kolone = [izbacene_kolone, poslednje_izbacena];
        izbaceni_indeksi = [izbaceni_indeksi, indeks_najslabije];

        X_trenutno(:, indeks_najslabije) = [];
        izabrane_kolone(indeks_najslabije) = [];

        beta_novi = (X_trenutno' * X_trenutno) \ (X_trenutno' * y_izlaz);
        rss_novi = rss(X_trenutno, y_izlaz, beta_novi);

        sigma2 = rss_novi / (N - size(X_trenutno, 2));
        varijansa_beta = diag(inv(X_trenutno' * X_trenutno) * sigma2);
        z_score = beta_novi ./ sqrt(varijansa_beta);

        broj_izbacenih = length(izbaceni_indeksi);
        F_score = ((rss_novi - rss_stari) / broj_izbacenih) / (rss_stari / (N - D));

        kriticna_vrednost = chi2inv(0.95, broj_izbacenih) / broj_izbacenih;
    end

    izabrane_kolone = sort([izabrane_kolone, poslednje_izbacena]);

    X_konacno = X_ulaz(:, izabrane_kolone);

    beta_konacno = (X_konacno' * X_konacno) \ (X_konacno' * y_izlaz);

    sigma2 = sum((y_izlaz - X_konacno * beta_konacno).^2) / (N - length(izabrane_kolone));
    varijansa_beta = diag(inv(X_konacno' * X_konacno) * sigma2);
    z_score = beta_konacno ./ sqrt(varijansa_beta);

    disp("Optimalne kolone dobijene backward stepwise selekcijom su:");
    disp(izabrane_kolone);

end

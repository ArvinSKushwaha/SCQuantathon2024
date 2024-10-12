"""
Helper Functions
----------------
This script conains functions needed to generate objects or
processes within the main script.
"""

import numpy as np
from numpy.typing import NDArray
import scipy as sp
from scipy.sparse import lil_array
from copy import deepcopy


def shuffle_vecs(vectors: NDArray) -> tuple[list, list]:
    """
    Shuffles an array of vectors. This function is used on the collection of basis vectors that will be used to
    build the subspace. Shuffling ensures a random process to how the vectors with high overlap are removed. This
    can be repeated until the optimal elimination process occurs.

    Parameters:
    -----------
    vectors : `NDArray`
        The set of vectors to be shuffled.

    Returns:
    --------
    shuffled_vecs : `list`
        The same set of vectors, now with their order shuffled.
    """
    shuffled_vecs = []
    choices = list(range(len(vectors)))
    new_order = []
    for i in range(len(choices)):
        choice = np.random.choice(choices)
        new_order.append(choice)
        shuffled_vecs.append(vectors[choice])
        choices.remove(choice)

    return shuffled_vecs, new_order


def get_overlap_matrix(basis: NDArray) -> NDArray:
    """
    Produces the overlap matrix.

    Parameters:
    -----------
    basis : `NDArray`
        The set of basis vectors.

    Returns:
    overlap : `NDArray`
        The overlap matrix.
    """
    nvecs = len(basis)
    overlap = np.zeros((nvecs, nvecs), dtype=complex)

    if type(basis[0]) == lil_array:

        for i in range(nvecs):
            ui = basis[i].tocsr()

            for j in range(i, nvecs):
                uj = basis[j].tocsr()

                overlap[i, j] = (ui.conj().T @ uj).toarray()[0][0]

                if not i == j:
                    overlap[j, i] = np.conjugate(overlap[i, j])
    else:

        for i in range(nvecs):
            ui = basis[i, :]

            for j in range(i, nvecs):
                uj = basis[j, :]

                overlap[i, j] = ui.conj().T @ uj

                if not i == j:
                    overlap[j, i] = np.conjugate(overlap[i, j])

    return overlap


def remove_high_overlap(
    arrays: NDArray, det_thresh=1e-6, condition_thresh=1e6, num_of_states="any"
) -> tuple[NDArray, list]:
    """
    Removes vectors from the set 'arrays' which have high overlap with other vectors in the set.

    Parameters:
    -----------
    arrays : `NDArray`
        The set of vectors to be reduced.
    det_thresh : `float` (Optional)
        Since the set of vectors will be used to build the overlap matrix, to ensure the matrix remains invertable its
        determinant must not go to zero. The determinant threshold enforces this with some tolerance. Defaults to 1e-6.
    condition_thresh : `float` (Optional)
        Similar to 'det_thresh', this ensures the resulting overlap matrix from the reduced set of vectors is valid by
        enforcing the condition number remain small. Defaults to 1e3.

    Returns:
    --------
    basis : `NDArray`
        The reduced set of vectors to be used to build the basis for the subspace.
    """
    basis = []
    remaining_order = []
    arrays, order = shuffle_vecs(arrays)

    for i, vec in enumerate(arrays):
        basis.append(vec)
        basis = np.array(basis)
        if num_of_states == 1:
            break
        else:
            if i > 2:
                overlap = get_overlap_matrix(basis)
                det = np.linalg.det(overlap)
                condition = np.linalg.cond(overlap)
                if det < det_thresh:
                    basis = basis[:-1]
                elif condition > condition_thresh:
                    basis = basis[:-1]
                else:
                    remaining_order.append(order[i])

            else:
                remaining_order.append(order[i])

            basis = list(basis)

            if num_of_states != "any":
                if len(basis) == num_of_states:
                    break

    return np.array(basis), remaining_order


def get_reduced_hamiltonian(hamiltonian: NDArray, basis: NDArray) -> NDArray:
    """
    Produces the reduced Hamiltonian `H_ij` for eigenvector continuation.

    Parameters:
    -----------
    hamiltonian : `NDArray`
        The hamiltonian to be reduced.
    basis : `NDArray`
        The basis vectors used to reduce the 'hamiltonian' matrix.

    Reurns:
    -------
    reduced_Ham : `NDArray`
        The reduced hamiltonian.
    """
    nvecs = len(basis)
    reduced_Ham = np.zeros((nvecs, nvecs), dtype="complex")

    if type(basis[0]) == lil_array:

        for i in range(nvecs):
            ui = basis[i].tocsr()

            for j in range(i, nvecs):
                uj = basis[j].tocsr()

                reduced_Ham[i, j] = (ui.conj().T @ hamiltonian @ uj).toarray()[0][0]

                if not i == j:
                    reduced_Ham[j, i] = np.conjugate(reduced_Ham[i, j])
    else:

        for i in range(nvecs):
            ui = basis[i, :]

            for j in range(i, nvecs):
                uj = basis[j, :]

                reduced_Ham[i, j] = np.conjugate(np.transpose(ui)) @ hamiltonian @ uj

                if not i == j:
                    reduced_Ham[j, i] = np.conjugate(reduced_Ham[i, j])

    return reduced_Ham


def get_fidelity(vector1: NDArray, vector2: NDArray) -> float:
    """
    Calculates the fidelity between two vectors, defined as | <phi_1|phi_2> |^2.

    Parameters:
    -----------
    vector1 : `NDArray`
        The first vector.
    vector2 : `NDArray`
        The second vector.

    Returns:
    --------
    fidelity : `float`
        The fidelity between the two vectors.
    """
    if type(vector1) == np.ndarray:
        fidelity = np.absolute(np.dot(vector1, vector2)) ** 2
    else:
        fidelity = np.absolute((vector1.conj().T @ vector2).toarray()[0][0]) ** 2
    return fidelity


def pretty_print(vector: NDArray):
    """
    Prints a vector or an array of vectors in a human readable format.

    Parameters:
    -----------
    vector : `NDArray`
        The vector or array of vectors to print.
    """
    if type(vector[0]) == NDArray:
        for vec in vector:
            nbits = int(np.log2(len(vec)))

            formatstr = "{0:>0" + str(nbits) + "b}"

            ix = -1
            for x in vec:
                ix += 1
                if abs(x) < 1e-6:
                    continue
                else:
                    print(formatstr.format(ix), ": ", np.round(x, 4))
            print("----------------------------")
    else:
        nbits = int(np.log2(len(vector)))

        formatstr = "{0:>0" + str(nbits) + "b}"

        ix = -1
        for x in vector:
            ix += 1
            if abs(x) < 1e-6:
                continue
            else:
                print(formatstr.format(ix), ": ", np.round(x, 4))
        print("----------------------------")


def get_gs_en_from_vec(vecs, ham):
    ens = []
    for vec in vecs:
        ens.append((vec.conj().T @ ham @ vec).real)
    return np.array(ens)


def test_subspace_diagonalization_with_slices(
    small_H, small_overlap, subspace_vectors, big_H
):

    evals, evecs = sp.linalg.eigh(small_H, small_overlap)
    nvecs = len(subspace_vectors)

    if type(subspace_vectors[0]) == lil_array:
        EC_ground_state = lil_array(subspace_vectors[0].shape, dtype=complex)
    else:
        EC_ground_state = np.zeros_like(subspace_vectors[0])

    for l in range(nvecs):
        EC_ground_state += evecs[l, 0] * subspace_vectors[l]

    if type(subspace_vectors[0]) == lil_array:
        EC_ground_state = EC_ground_state / sp.sparse.linalg.norm(EC_ground_state)
    else:
        EC_ground_state = EC_ground_state / np.linalg.norm(EC_ground_state)

    if np.linalg.norm(evecs[:, 0]) < 10:
        EC_ground_energy = evals[0]
        coeff_flag = False
        ec_coeff_norm = np.linalg.norm(evecs[:, 0])
    else:
        coeff_flag = True
        ec_coeff_norm = np.linalg.norm(evecs[:, 0])

        if type(subspace_vectors[0]) == lil_array:
            EC_ground_energy = (
                (EC_ground_state.conj().T @ big_H @ EC_ground_state)
                .toarray()[0][0]
                .real
            )
        else:
            EC_ground_energy = (EC_ground_state.conj().T @ big_H @ EC_ground_state).real

    return EC_ground_energy, EC_ground_state, coeff_flag, ec_coeff_norm, evecs[:, 0]


def optimize_subspace(
    subspace,
    reduced_hamiltonian,
    overlap_matrix,
    exact_Hamiltonian,
    exact_energy,
    exact_vector,
    convergence_threshold: float = 1e-6,
    discard_threshold: float = 1e-3,
    optimize_amplitudes: bool = False,
    verbose: bool = False,
):

    all_ec_ens = []
    all_ec_vecs = []
    all_ec_coeffs = []
    fidelities = []
    dont_add = set()
    chosen = []
    not_chosen = set(range(len(subspace)))

    num_it = len(subspace) - 1
    num_choices = len(subspace)
    N_sites = int(np.log2(exact_Hamiltonian.shape[0]))

    next_choice = np.argwhere(
        reduced_hamiltonian == np.min(np.diag(reduced_hamiltonian))
    )[0][0]
    all_ec_vecs.append(deepcopy(subspace[next_choice]))
    all_ec_ens.append(reduced_hamiltonian[next_choice][next_choice].real)
    all_ec_coeffs.append(np.array([1]))
    chosen.append(next_choice)
    not_chosen.remove(next_choice)
    fidelities.append(get_fidelity(all_ec_vecs[-1], exact_vector))

    # st = "\n\n### 1 state ###"
    st = f"\nGaussian state {next_choice} has min energy {reduced_hamiltonian[next_choice][next_choice].real} (exact is {exact_energy})"
    st += f"\nFidelity: {fidelities[-1]}"
    if verbose:
        print(st)

    for i in range(num_it):

        if N_sites > 8:
            if i == 0:
                # overlaps = np.array([(all_ec_vecs[-1].T @ vec).toarray()[0][0] for vec in subspace[list(not_chosen)]])
                overlaps = overlap_matrix[chosen, list(not_chosen)]
            else:
                overlaps = np.array(
                    [
                        sum(
                            [
                                all_ec_coeffs[-1][ix] * overlap_matrix[k, chosen[ix]]
                                for ix in range(len(chosen))
                            ]
                        )
                        for k in not_chosen
                    ]
                )
        else:
            if i == 0:
                # overlaps = np.array([(all_ec_vecs[-1].conj().T @ vec).real for vec in subspace[list(not_chosen)]])
                overlaps = overlap_matrix[chosen, list(not_chosen)]
            else:
                overlaps = np.array(
                    [
                        sum(
                            [
                                all_ec_coeffs[-1][ix] * overlap_matrix[k, chosen[ix]]
                                for ix in range(len(chosen))
                            ]
                        )
                        for k in not_chosen
                    ]
                )

        overlaps_ixs = np.array([ix for ix in range(len(subspace)) if ix in not_chosen])
        next_choices = [ix for ix in overlaps_ixs[np.argsort(abs(overlaps))]]
        next_choices = [choice for choice in next_choices if choice not in dont_add][
            :num_choices
        ]
        potential_chosen = []
        potential_ec_energies = []
        potential_ec_vecs = []
        potential_ec_coeffs = []

        for choice in next_choices:
            potential_choices = deepcopy(chosen) + [choice]
            small_overlap = overlap_matrix[potential_choices][:, potential_choices]
            small_ham = reduced_hamiltonian[potential_choices][:, potential_choices]
            chosen_vecs = subspace[potential_choices]
            try:
                ec_energy, ec_vec, flag, norm, ec_coeffs = (
                    test_subspace_diagonalization_with_slices(
                        small_ham, small_overlap, chosen_vecs, exact_Hamiltonian
                    )
                )
                potential_ec_energies.append(ec_energy)
                potential_ec_vecs.append(ec_vec)
                potential_ec_coeffs.append(ec_coeffs)
                potential_chosen.append(potential_choices)

                # if ec_energy < all_ec_ens[-1]:
                #     break
            except:
                dont_add.add(choice)
                not_chosen.remove(choice)
                continue

        if len(potential_ec_energies) == 0:
            continue
        # if np.min(potential_ec_energies) > all_ec_ens[-1]:
        #     dont_add.add(potential_chosen[np.argmin(potential_ec_energies)][-1])
        #     not_chosen.remove(potential_chosen[np.argmin(potential_ec_energies)][-1])
        #     continue

        choice = potential_chosen[np.argmin(potential_ec_energies)][-1]
        chosen.append(choice)
        not_chosen.remove(choice)

        fidelities.append(get_fidelity(all_ec_vecs[-1], exact_vector))
        all_ec_ens.append(potential_ec_energies[np.argmin(potential_ec_energies)])
        all_ec_vecs.append(potential_ec_vecs[np.argmin(potential_ec_energies)])
        all_ec_coeffs.append(potential_ec_coeffs[np.argmin(potential_ec_energies)])

        if verbose:
            print(f"\n### {len(all_ec_ens)} states ###")
            print(
                f"Adding gaussian state {choice} gives energy {all_ec_ens[-1]} (exact is {exact_energy})"
            )
            print(
                f"Relative energy difference: {abs((all_ec_ens[-1]-exact_energy)/exact_energy)}"
            )
            print(f"Fidelity: {get_fidelity(all_ec_vecs[-1],exact_vector)}")
            print(f"Choices left: {len(not_chosen)}")
            print(f"Ill-conditioned choices: {len(dont_add)}")
            print(f"Chosen: {chosen}")
            print(f"Num iterations left: {num_it-i-1}")

        if (
            len(
                [
                    choice
                    for choice in np.argsort(abs(overlaps))
                    if choice not in dont_add
                ]
            )
            == 0
        ):
            break

        if abs(all_ec_ens[-1] - all_ec_ens[-2]) < convergence_threshold:
            if verbose:
                print("Converged!")
            break

    if verbose:
        print("\n\n### Subspace optimized ###")
        print(f"Number of states: {len(all_ec_ens)}")
        print(
            f"Energy difference with exact: {abs((all_ec_ens[-1]-exact_energy)/exact_energy)}"
        )
        print(f"Fidelity with exact: {get_fidelity(all_ec_vecs[-1],exact_vector)}")
        print(f"EC coefficient norm: {np.linalg.norm(all_ec_coeffs[-1])}")

        print("\n### Removing states with small coefficients ###")
    smallest_coeffs_ix = [
        i
        for i in range(len(all_ec_coeffs))
        if abs((all_ec_coeffs[-1][i]) / np.max(all_ec_coeffs[-1])) < discard_threshold
    ]
    new_chosen = [chosen[c] for c in range(len(chosen)) if c not in smallest_coeffs_ix]
    small_overlap = overlap_matrix[new_chosen][:, new_chosen]
    small_ham = reduced_hamiltonian[new_chosen][:, new_chosen]
    chosen_vecs = subspace[new_chosen]

    # import matplotlib.pyplot as plt
    # plt.imshow(small_overlap.real)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(overlap_matrix[chosen][:, chosen].real)
    # plt.colorbar()
    # plt.show()

    try:
        ec_energy, ec_vec, flag, norm, new_ec_coeffs = (
            test_subspace_diagonalization_with_slices(
                small_ham, small_overlap, chosen_vecs, exact_Hamiltonian
            )
        )
        if verbose:
            print(f"\n### Final subspace ###")
            print(
                f"Basis reduced from {len(all_ec_coeffs[-1])} to {len(new_ec_coeffs)} states"
            )
            print(
                f"Energy difference with old basis: {abs((all_ec_ens[-1]-exact_energy)/exact_energy)}"
            )
            print(
                f"Energy difference with new basis: {abs((ec_energy-exact_energy)/exact_energy)}"
            )

            print(
                f"Fidelity with old basis: {get_fidelity(all_ec_vecs[-1],exact_vector)}"
            )
            print(f"Fidelity with new basis: {get_fidelity(ec_vec,exact_vector)}")

            print(
                f"EC coefficient norm with old basis: {np.linalg.norm(all_ec_coeffs[-1])}"
            )
            print(
                f"EC coefficient norm with new basis: {np.linalg.norm(new_ec_coeffs)}"
            )

    except:
        if verbose:
            print(
                "\nReducing the subspace is ill-conditioned.\n ... Proceeding with the full optimal subspace ..."
            )
        new_chosen = chosen
        new_ec_coeffs = all_ec_coeffs[-1]
        ec_energy = all_ec_ens[-1]
        ec_vec = all_ec_vecs[-1]

    selected = subspace[new_chosen]
    amplitudes = new_ec_coeffs.real / np.linalg.norm(new_ec_coeffs.real)

    if optimize_amplitudes is True:

        def get_energy(vec, ham):
            N_sites = int(np.log2(vec.shape[0]))
            if N_sites > 8:
                return (vec.T @ ham @ vec).toarray().real[0][0]
            else:
                return (vec.conj().T @ ham @ vec).real

        def build_lincomb(vecs, amplitudes):
            N_sites = int(np.log2(vecs[0].shape[0]))

            if N_sites > 8:
                lincomb_vec = lil_array((vecs[0].shape[0], 1), dtype=complex)
            else:
                lincomb_vec = np.zeros(len(vecs[0]), dtype=complex)

            for a, vec in enumerate(vecs):
                lincomb_vec += amplitudes[a] * vec

            if N_sites > 8:
                lincomb_vec = lincomb_vec / sp.sparse.linalg.norm(lincomb_vec)
            else:
                lincomb_vec = lincomb_vec / np.linalg.norm(lincomb_vec)  # type: ignore

            return lincomb_vec

        def cost(amplitudes, vecs, ham):
            lincomb_vec = build_lincomb(vecs, amplitudes)
            return get_energy(lincomb_vec, ham)

        if verbose:
            print("\n\n### Optimizing amplitudes ###")

        res = sp.optimize.minimize(
            cost,
            amplitudes,
            args=(selected, exact_Hamiltonian),
            options={"maxiter": 100},
            bounds=[(-10, 10)] * len(selected),
            tol=1e-6,
        )
        best_vec = build_lincomb(selected, res.x)
        best_en = get_energy(best_vec, exact_Hamiltonian)

        # print(st)
        # log.append(st)
        # update_log(log, log_file)

        if verbose:
            print("\n\n### Final optimized state ###")
            st = f"Best energy: {get_energy(best_vec,exact_Hamiltonian)}"
            st += f"\nRelative energy difference: {abs((get_energy(best_vec,exact_Hamiltonian)-exact_energy)/exact_energy)}"
            st += f"\nBest fidelity: {get_fidelity(best_vec,exact_vector)}"
            st += f"\nCoefficient norm: {np.linalg.norm(res.x)}"
            print(st)
        # log.append(st)
        # update_log(log, log_file)

        # return ec_energy, ec_vec, get_gs_en_from_vec(subspace, exact_Hamiltonian), chosen

        return (
            best_en,
            best_vec,
            get_gs_en_from_vec(subspace, exact_Hamiltonian),
            chosen,
        )

    else:
        return (
            ec_energy,
            ec_vec,
            get_gs_en_from_vec(subspace, exact_Hamiltonian),
            chosen,
        )
